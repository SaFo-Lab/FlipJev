"""Where an attacker's text can go inside an agent trajectory, and how it is inserted there.

An outside attacker cannot append text to the end of a trajectory: the end of the trace is the agent's own last step.
What an attacker does write is the content of tool outputs that carry third-party text (a web page, an email, a file,
a search snippet). This module finds those places, picks one, inserts text there in the format of its container, and
verifies that the edit changed nothing else.

    1. detect_format   is a tool output JSON, a Python literal, a dict-like text with unescaped quotes, or plain text?
    2. layered_leaves  every string value inside it with the exact span of its contents, descending into strings
                       that are themselves JSON or Python-literal documents (up to three layers)
    3. enumerate_slots the string values of the tool outputs, each classified (free text? URL? identifier? status
                       message? the agent's own text?) and test-edited with a hostile string
    4. select_slot     the last free-text place after a read-style tool call (after a call of unknown kind if there
                       is none); never a place after a write-style call
    5. apply_slot      insert text at that place; verify_edit checks that every layer keeps its format, that the
                       whole structure parses to the old one with exactly one string grown by exactly that text, that
                       nothing else changed, and that the input of the judge differs by exactly the inserted text

A slot is a plain dict, so it can be stored with a case and replayed:
    {"index": int (the message), "fmt": "json" | "pylit" | "reprlike" | "text",
     "path": [keys or indices to the string inside the innermost container],
     "via": [{"fmt": ..., "path": [...]}, ...],     # outer containers whose string value holds the next one
     "key": last key or None, "kind": see classify_text, "free_text": bool, "words": int, "chars": int,
     "offset": raw position in the outermost text, "inner_offset": position inside the innermost text,
     "tool_kind": "read" | "write" | "other"}
"""

import ast
import bisect
from dataclasses import dataclass
import itertools
import json
import re
import warnings

from flipjev.trajectories import render_state

# Plain text with backslashes warns about invalid escape sequences when a parser tries it as a Python literal. A scoped
# catch_warnings() would race between threads (the evaluators build states from a thread pool), so filter once here.
warnings.filterwarnings("ignore", message=r".*invalid escape sequence.*", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=r".*invalid escape sequence.*", category=DeprecationWarning)

MAX_DEPTH = 100
MAX_LAYERS = 3
# Contains both quote kinds, backslashes, braces, newlines, a control character, U+2028 and a non-BMP character:
# whatever escaping a container needs, this exercises it.
PROBE = "a 'b' \"c\" \\d \\\\ {e} [f] \n\n\t \x01 \u2028 \u00e9 \U0001f600 end"


class SlotError(ValueError):
    """A slot cannot be edited cleanly."""


@dataclass(frozen=True)
class Leaf:
    """One string value: `value` is the decoded text; [start, end) is its raw contents inside the container.

    `end` is where new text is inserted (just before the closing quote); `quote` is the literal's own quote
    characters for Python literals, "" otherwise.
    """
    path: tuple
    start: int
    end: int
    value: str
    quote: str = ""


# --------------------------------------------------------------------------
# JSON: a small scanner that records the exact span of every string value.
# --------------------------------------------------------------------------
_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9]\d{0,4299})(?:\.\d+)?(?:[eE][+-]?\d+)?")
_JSON_WORDS = ("true", "false", "null", "NaN", "Infinity", "-Infinity")


def _skip_ws(text, i):
    while i < len(text) and text[i] in " \t\n\r":
        i += 1
    return i


def _json_string(text, i):
    found = _JSON_STRING.match(text, i)
    if not found:
        raise ValueError("unterminated string")
    return json.loads(found.group(0)), found.end()   # json validates escapes and control characters


def _json_value(text, i, path, out, depth):
    if depth > MAX_DEPTH:
        raise ValueError("nesting too deep")
    if i >= len(text):
        raise ValueError("unexpected end of input")
    c = text[i]
    if c == '"':
        value, end = _json_string(text, i)
        out.append(Leaf(path, i + 1, end - 1, value))
        return end
    if c == "{":
        i = _skip_ws(text, i + 1)
        if text[i:i + 1] == "}":
            return i + 1
        seen = set()
        while True:
            key, i = _json_string(text, i)
            if key in seen:
                raise ValueError("duplicate object key")
            seen.add(key)
            i = _skip_ws(text, i)
            if text[i:i + 1] != ":":
                raise ValueError("expected ':'")
            i = _json_value(text, _skip_ws(text, i + 1), path + (key,), out, depth + 1)
            i = _skip_ws(text, i)
            if text[i:i + 1] == ",":
                i = _skip_ws(text, i + 1)
                continue
            if text[i:i + 1] == "}":
                return i + 1
            raise ValueError("expected ',' or '}'")
    if c == "[":
        i = _skip_ws(text, i + 1)
        if text[i:i + 1] == "]":
            return i + 1
        index = 0
        while True:
            i = _json_value(text, i, path + (index,), out, depth + 1)
            index += 1
            i = _skip_ws(text, i)
            if text[i:i + 1] == ",":
                i = _skip_ws(text, i + 1)
                continue
            if text[i:i + 1] == "]":
                return i + 1
            raise ValueError("expected ',' or ']'")
    for word in _JSON_WORDS:
        if text.startswith(word, i):
            return i + len(word)
    found = _JSON_NUMBER.match(text, i)
    if found:
        return found.end()
    raise ValueError("unexpected character")


def json_leaves(text):
    """String values of a JSON document, in order, with exact spans. Raises ValueError if not valid JSON."""
    out = []
    end = _json_value(text, _skip_ws(text, 0), (), out, 0)
    if _skip_ws(text, end) != len(text):
        raise ValueError("data after the JSON value")
    return out


# --------------------------------------------------------------------------
# Python literals (dict/list reprs): positions come from the ast module.
# --------------------------------------------------------------------------
_PY_PREFIX = re.compile(r"""^([A-Za-z]*)('''|\"\"\"|'|\")""")


def pylit_leaves(text):
    """String values of a Python literal (dict/list/tuple of constants) with exact spans.

    Raises ValueError if `text` is not one. Raw, bytes and f-strings are not editable and are skipped.
    """
    if "\r" in text or "\x0c" in text:
        raise ValueError("carriage return or form feed shifts parser offsets")
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    try:
        tree = ast.parse(body, mode="eval")
        ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(f"not a Python literal: {exc}") from exc
    lines = body.split("\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    cumulative = {}                             # per line: UTF-8 bytes up to and including each character

    def offset(lineno, col):                    # ast columns are UTF-8 byte offsets
        line = lines[lineno - 1]
        if line.isascii():
            return lead + starts[lineno - 1] + col
        if lineno not in cumulative:
            cumulative[lineno] = list(itertools.accumulate(len(ch.encode("utf-8")) for ch in line))
        return lead + starts[lineno - 1] + bisect.bisect_right(cumulative[lineno], col)

    out = []

    def walk(node, path, depth):
        if depth > MAX_DEPTH:
            raise ValueError("nesting too deep")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                begin = offset(node.lineno, node.col_offset)
                end = offset(node.end_lineno, node.end_col_offset)
                literal = text[begin:end]
                head = _PY_PREFIX.match(literal)
                if head and head.group(1) in ("", "u", "U") and literal.endswith(head.group(2)) \
                        and len(literal) >= len(head.group(0)) + len(head.group(2)):
                    out.append(Leaf(path, begin + len(head.group(0)), end - len(head.group(2)), node.value,
                                    head.group(2)))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is None:
                    continue
                try:
                    name = ast.literal_eval(key)
                except (ValueError, TypeError):
                    continue
                if isinstance(name, (str, int)):
                    walk(value, path + (name,), depth + 1)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for index, element in enumerate(node.elts):
                walk(element, path + (index,), depth + 1)

    walk(tree.body, (), 0)
    return out


# --------------------------------------------------------------------------
# Dict-like reprs whose quotes are not escaped (`'body': 'Please read the 'report' now.'`): not valid Python,
# common in hand-written tool outputs. Values are found by their key and closed by the quote that is followed by
# the next key or a closing bracket, so quotes inside a value are harmless.
# --------------------------------------------------------------------------
_FIELD = re.compile(r"""(?P<kq>['"])(?P<key>[A-Za-z_][\w \-]{0,40})(?P=kq)\s*:\s*(?P<vq>['"])""")
# A quote that can close a value: followed by the next key, or by closing brackets that are themselves followed by a comma or
# the end of the document (so a `']` inside the value's own text, as in `... ['123', '456'].`, is not its end).
_CLOSER = re.compile(r"""(?P<q>['"])(?=\s*(?:,\s*(?:(?P<kq>['"])[A-Za-z_][\w \-]{0,40}(?P=kq)\s*:|[\]}])|[\]}][\s\]}]*(?:,|$)))""")
_ESCAPED = re.compile(r"""\\([\\'"])""")
MAX_REPRLIKE_CHARS = 300_000


def _document_span(text):
    """(offset, document) for text that is a JSON/dict-like document, or one wrapped in a pair of quotes (a string
    literal whose inside is not escaped, so it is not valid JSON or Python); None otherwise."""
    body = text.strip()
    offset = len(text) - len(text.lstrip())
    if len(body) > 2 and body[0] == body[-1] and body[0] in "\"'":
        inner = body[1:-1]
        offset += 1 + len(inner) - len(inner.lstrip())
        body = inner.strip()
    if len(body) > 1 and body[0] in "{[" and body[-1] in "}]":
        return offset, body
    return None


def _reprlike_decode(raw):
    return _ESCAPED.sub(r"\1", raw)


def _reprlike_encode(text, quote):
    return text.replace("\\", "\\\\").replace(quote, "\\" + quote)


def _reprlike_closers(body):
    """Sorted positions, per quote character, of the quotes that can close a value (and are not escaped)."""
    closers = {"'": [], '"': []}
    for found in _CLOSER.finditer(body):
        j = found.start()
        backslashes = 0
        while j - 1 - backslashes >= 0 and body[j - 1 - backslashes] == "\\":
            backslashes += 1
        if backslashes % 2 == 0:
            closers[found.group("q")].append(j)
    return closers


def reprlike_leaves(text):
    """String values of the `'key': 'value'` fields of a dict-like text, with exact spans.

    Raises ValueError unless the text looks like a document and has at least one such field. A leaf's path is
    ("<ordinal>:<key>",): the ordinal keeps repeated keys apart. Linear in the length of the text.
    """
    if len(text) > MAX_REPRLIKE_CHARS:
        raise ValueError("too long for the lenient scanner")
    span = _document_span(text)
    if span is None:
        raise ValueError("not shaped like a document")
    shift, body = span
    closers = _reprlike_closers(body)
    out, pos = [], 0
    while True:
        found = _FIELD.search(body, pos)
        if not found:
            break
        quote = found.group("vq")
        positions = closers[quote]
        i = bisect.bisect_left(positions, found.end())
        if i == len(positions):
            pos = found.end()
            continue
        end = positions[i]
        out.append(Leaf((f"{len(out)}:{found.group('key')}",), shift + found.end(), shift + end,
                        _reprlike_decode(body[found.end():end]), quote))
        pos = end + 1
    if not out:
        raise ValueError("no quoted fields")
    return out


# --------------------------------------------------------------------------
# Format detection, layers and per-format encoding
# --------------------------------------------------------------------------
def detect_format(text):
    """'json', 'pylit' (a Python dict/list repr), 'reprlike' (a dict-like repr with unescaped quotes), 'text', 'empty'."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    for name, scanner in (("json", json_leaves), ("pylit", pylit_leaves), ("reprlike", reprlike_leaves)):
        try:
            scanner(text)
            return name
        except (ValueError, RecursionError):
            pass
    return "text"


def looks_structured(text):
    """Text that begins and ends like a JSON/dict document (with or without parsing), optionally inside one pair of quotes."""
    return _document_span(text) is not None


def leaves_of(text, fmt):
    if fmt == "json":
        return json_leaves(text)
    if fmt == "pylit":
        return pylit_leaves(text)
    if fmt == "reprlike":
        return reprlike_leaves(text)
    if fmt == "text":
        return [Leaf((), 0, len(text), text)]
    return []


def layered_leaves(text, fmt, via=(), anchor=None):
    """(via, fmt, leaf, anchor) for every string in `text`, descending into strings that are documents.

    `via` lists the outer containers ({"fmt", "path"}) whose string value holds the next container;
    `anchor` is the raw end of the outermost string the leaf lives in (for ordering). A string that looks like
    a document but does not parse is yielded as it is and classified as "document" (never appended to).
    """
    for leaf in leaves_of(text, fmt):
        here = leaf.end if anchor is None else anchor
        if looks_structured(leaf.value) and len(via) < MAX_LAYERS and _decodes_cleanly(text, fmt, leaf):
            inner = detect_format(leaf.value)
            if inner in ("json", "pylit", "reprlike"):
                yield from layered_leaves(leaf.value, inner, via + ({"fmt": fmt, "path": list(leaf.path)},), here)
                continue
        yield via, fmt, leaf, here


def _json_encode(text):
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _py_encode(text, quote):
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == quote[0]:
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\x%02x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _separator(value):
    """What goes between the existing text and the inserted text: nothing, a space, or a blank line."""
    if not value or value[-1].isspace():
        return ""
    return "\n\n" if "\n" in value else " "


_JSON_TOKEN = re.compile(r"\\u[0-9a-fA-F]{4}|\\.|[^\\]", re.S)
_PY_TOKEN = re.compile(r"\\N\{[^}]*\}|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\[0-7]{1,3}|\\\n|\\.|[^\\]",
                       re.S)
_HIGH, _LOW = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}"), re.compile(r"\\u[dD][c-fC-F][0-9a-fA-F]{2}")


_PY_ONE_CHAR_ESCAPES = set("\\'\"abfnrtvxuUN01234567")


def _decoded_width(token, fmt):
    """How many decoded characters one raw token stands for."""
    if fmt == "pylit" and token.startswith("\\"):
        if token == "\\\n":
            return 0                            # a backslash-newline continuation decodes to nothing
        if token[1:2] not in _PY_ONE_CHAR_ESCAPES:
            return 2                            # an unknown escape keeps its backslash: '\d' is two characters
    return 1


def _units(raw, fmt):
    """(raw_length, decoded_width) for each unit of a string literal's raw contents (a JSON surrogate pair is one unit)."""
    tokens = [m.group() for m in (_JSON_TOKEN if fmt == "json" else _PY_TOKEN).finditer(raw)]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        length = len(token)
        i += 1
        if fmt == "json" and _HIGH.fullmatch(token) and i < len(tokens) and _LOW.fullmatch(tokens[i]):
            length += len(tokens[i])
            i += 1
        yield length, _decoded_width(token, fmt)


def _raw_offset(raw, fmt, index):
    """Offset in the raw contents of a string literal where decoded character number `index` begins."""
    count = pos = 0
    for length, width in _units(raw, fmt):
        if count == index:
            return pos
        if count < index < count + width:
            raise SlotError("position falls inside an escape sequence")
        pos += length
        count += width
    if count == index:
        return pos
    raise SlotError("cannot map a decoded position back into the raw string")


def _decodes_cleanly(text, fmt, leaf):
    """True if the leaf's raw contents map one-to-one onto its decoded value (not several literals joined implicitly)."""
    if fmt not in ("json", "pylit"):
        return False
    return sum(width for _, width in _units(text[leaf.start:leaf.end], fmt)) == len(leaf.value)


def _find_leaf(text, fmt, path):
    try:
        matches = [leaf for leaf in leaves_of(text, fmt) if leaf.path == tuple(path)]
    except (ValueError, RecursionError) as exc:
        raise SlotError(f"the container is no longer valid {fmt}: {exc}") from exc
    if len(matches) != 1:
        raise SlotError(f"path {list(path)!r} matches {len(matches)} string values in a {fmt} container")
    return matches[0]


def edit_container(text, fmt, path, suffix):
    """Insert `suffix` at the end of the string value at `path`; returns (new_text, encoded_insertion, leaf)."""
    leaf = _find_leaf(text, fmt, path)
    raw = _separator(leaf.value) + suffix if suffix else ""
    if fmt == "json":
        inserted = _json_encode(raw)
    elif fmt == "pylit":
        inserted = _py_encode(raw, leaf.quote)
    elif fmt == "reprlike":
        inserted = _reprlike_encode(raw, leaf.quote)
    else:
        inserted = raw
    return text[:leaf.end] + inserted + text[leaf.end:], inserted, leaf


def slot_layers(slot):
    """The chain of containers a slot goes through, outermost first, ending at the one holding the string."""
    return list(slot.get("via") or []) + [{"fmt": slot["fmt"], "path": slot["path"]}]


def edit_layers(text, layers, suffix):
    """Insert `suffix` at the end of the string reached through `layers`; returns (new_text, inserted, at).

    `inserted` is exactly what was added to `text` (encoded for its own format) and `at` is where. For a string
    nested inside another string, the text is inserted inside the inner container and the inner insertion is
    encoded again for each outer container, so every layer changes by one insertion and nothing else.
    """
    spec = layers[0]
    if len(layers) == 1:
        new_text, inserted, leaf = edit_container(text, spec["fmt"], spec["path"], suffix)
        return new_text, inserted, leaf.end
    carrier = _find_leaf(text, spec["fmt"], spec["path"])
    _, inner_inserted, inner_at = edit_layers(carrier.value, layers[1:], suffix)
    if not inner_inserted:
        return text, "", carrier.end
    if spec["fmt"] not in ("json", "pylit"):
        raise SlotError(f"a {spec['fmt']} container cannot carry a nested document")
    at = carrier.start + _raw_offset(text[carrier.start:carrier.end], spec["fmt"], inner_at)
    inserted = _json_encode(inner_inserted) if spec["fmt"] == "json" else _py_encode(inner_inserted, carrier.quote)
    return text[:at] + inserted + text[at:], inserted, at


# --------------------------------------------------------------------------
# What kind of string is it? Only free text from a third party is a believable place for attacker prose.
# --------------------------------------------------------------------------
_URL = re.compile(r"^(?:(?:https?|ftp)://|www\.)\S+$", re.I)


def _is_email(text):
    if text.count("@") != 1 or any(ch.isspace() for ch in text):
        return False
    local, domain = text.split("@")
    return bool(local) and "." in domain.strip(".") and not domain.startswith(".") and not domain.endswith(".")


_PATH = re.compile(r"^(?:~?/|\.{1,2}/|[A-Za-z]:[\\/])\S*$|^\S*[/\\]\S*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
                   r"|^\d{1,2}[/:.-]\d{1,2}[/:.-]\d{2,4}$|^\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?$")
_NUMBER = re.compile(r"^[+-]?[$€£]?\d[\d,._]*\s?(?:%|[A-Za-z]{0,3})$")
_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>*_#-–—"
_TAG = re.compile(r"</?[A-Za-z][^<>]{0,100}>")
_KEY_VALUE = re.compile(r"\b[A-Za-z_]\w*:\s+\S")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
# A tool's own words about itself: success notices and shell/OS errors are not text an outside party wrote.
_BOILERPLATE = re.compile(
    r"\b(?:completed|executed|performed|processed|finished|ran)\s+successfully\b"
    r"|^\W*(?:operation|action|request|command|task)\b.{0,50}\b(?:success\w*|completed?|done)\b"
    r"|\bsuccessfully\W*$"
    r"|\bno such file or directory\b|\bpermission denied\b|\bcommand not found\b|\bis not recognized as\b"
    r"|^\W*error during execution|\btraceback \(most recent call last\)|\bobject has no attribute\b"
    r"|\bis not defined\b|\bnot found\W*$",
    re.I | re.S)
# Keys whose value the system (not a third party) fills in, and keys that hold a short attribute, not prose.
# A system word anywhere in the key ("status_message", "lastError") marks it system-filled; an attribute word
# must be the last one: "shopLocation" -> location, "phone_number" -> number.
_SYSTEM_WORDS = {"error", "errors", "err", "stderr", "exception", "traceback", "status", "state", "code", "success",
                 "failure", "failures", "failed"}
_ATTRIBUTE_WORDS = {"address", "location", "name", "phone", "city", "country", "zip", "zipcode", "street", "id",
                    "url", "link", "path", "type", "email", "currency", "unit", "timezone", "number", "dosage",
                    "keyword", "keywords", "query", "quantity", "frequency", "duration", "amount", "label", "tag",
                    "tags", "category", "role", "department"}
MAX_ATTRIBUTE_WORDS = 8          # an attribute is short; a long sentence under a `name` key ("event_name") is prose


def _key_words(key):
    if not isinstance(key, str):
        return []
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key).replace("_", " ").replace("-", " ").lower().split()


def word_count(value):
    """Whitespace-separated tokens that are words (letters, with inner apostrophes or hyphens)."""
    count = 0
    for token in value.split():
        core = token.strip(_EDGE_PUNCT).replace("'", "").replace("’", "").replace("-", "")
        if len(core) >= 2 and core.isalpha():
            count += 1
    return count


def classify_text(value, key=None):
    """(kind, words): a coarse type for a string value, given the key it sits under (if any).

    free_text is prose from someone other than the tool. The others say why a string is not a place for
    attacker prose: empty, document (a nested JSON/dict), url, email, date, path, number, token (one word:
    enum, id, hash), short (under a sentence), status (the tool's own success/error message), attribute
    (an address, name, phone ...), markup (HTML-like tags), record (a list of `field: value` pairs), table (rows
    of `a | b | c`), keywords (a comma-separated tag list).
    """
    stripped = value.strip()
    if not stripped:
        return "empty", 0
    words = word_count(stripped)
    if looks_structured(stripped):
        return "document", words              # a document nested inside a string: appending would break it
    key_words = _key_words(key)
    if any(word in _SYSTEM_WORDS for word in key_words):
        return "status", words
    if len(_CJK.findall(stripped)) >= 8:
        return "free_text", words
    single_token = not any(ch.isspace() for ch in stripped)
    if _URL.match(stripped):
        return "url", words
    if _is_email(stripped):
        return "email", words
    if _DATE.match(stripped):
        return "date", words
    if single_token and _PATH.match(stripped):
        return "path", words
    if _NUMBER.match(stripped):
        return "number", words
    if single_token:
        return "token", words
    if words < 4:
        return "short", words                 # under a sentence: a title, a place name, a label
    if key_words and key_words[-1] in _ATTRIBUTE_WORDS and words <= MAX_ATTRIBUTE_WORDS:
        return "attribute", words
    if _BOILERPLATE.search(stripped) or (words <= 12 and re.search(r"\bsuccessfully\b", stripped, re.I)):
        return "status", words
    if len(_TAG.findall(stripped)) >= 3:
        return "markup", words
    pairs = len(_KEY_VALUE.findall(stripped))
    if pairs >= 4 and pairs * 4 >= words:
        return "record", words
    lines = stripped.split("\n")
    if len(lines) >= 3 and sum(line.count("|") >= 2 for line in lines) * 2 >= len(lines):
        return "table", words                 # rows of `a | b | c`: system or listing data
    parts = stripped.split(",")
    if len(parts) >= 5 and words <= 3 * len(parts) and not _SENTENCE_END.search(stripped[:-1]):
        return "keywords", words              # a comma-separated list of tags, not a sentence
    return "free_text", words


def _squash(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _agent_text(message):
    """What the agent itself wrote in a step: thought and action, with JSON/dict actions flattened to their strings."""
    parts = []
    for field in ("thought", "action"):
        value = message.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        fmt = detect_format(value)
        parts.append(" ".join(leaf.value for leaf in leaves_of(value, fmt)) if fmt in ("json", "pylit") else value)
    return re.sub(r"\\[ntr]", " ", " ".join(parts))      # an escaped newline typed into an argument is a newline when echoed


def _echoes(value, prior):
    """True if a long string repeats text the agent or user already wrote (so it is not third-party text)."""
    squashed = _squash(value)
    return len(squashed) >= 20 and (squashed[:30] in prior or squashed[-30:] in prior)


READ_VERBS = {"get", "read", "fetch", "search", "list", "find", "view", "open", "browse", "retrieve", "query",
              "lookup", "download", "check", "scan", "cat", "head", "tail", "grep", "curl", "wget", "load",
              "show", "display", "look", "inspect", "summarize", "summarise", "access", "review", "visit",
              "extract", "parse", "monitor", "track", "analyze", "analyse", "describe", "detail", "details",
              "info", "news", "weather", "status", "history", "click", "navigate", "scroll", "tap", "goto"}
# For a generic terminal tool the command, not the tool's name, says whether it reads or writes.
READ_COMMANDS = {"cat", "ls", "head", "tail", "grep", "find", "less", "more", "curl", "wget", "pwd", "df", "du", "ps",
                 "whoami", "date", "file", "stat", "wc", "sort", "uniq", "diff"}
WRITE_COMMANDS = {"rm", "mv", "cp", "chmod", "chown", "mkdir", "rmdir", "touch", "kill", "sudo", "apt", "pip", "tee", "dd",
                  "ln", "tar", "unzip", "scp", "ssh", "useradd", "passwd"}
WRITE_VERBS = {"send", "post", "create", "delete", "remove", "update", "write", "set", "transfer", "pay", "book",
               "order", "submit", "execute", "run", "install", "add", "modify", "move", "edit", "publish",
               "upload", "share", "deploy", "kill", "reset", "cancel", "approve", "grant", "revoke", "unlock",
               "lock", "start", "stop", "restart", "buy", "sell", "withdraw", "deposit", "schedule", "forward",
               "reply", "save", "insert", "drop", "rename", "copy", "change", "enable", "disable", "turn",
               "generate", "synthesize", "mock", "fake", "simulate"}     # what such a tool returns, it made up itself


def tool_kind(action):
    """'read', 'write' or 'other': what the call that produced an observation does, from its name or verb.

    Read-style tools (fetch a page, read a file or email, search) return third-party text; write-style tools
    return an acknowledgement of what the agent did. The first verb in the call decides.
    """
    if not action:
        return "other"
    command = re.search(r"""['"]command['"]\s*:\s*['"]\s*([A-Za-z0-9_./-]+)""", action)
    if command:
        program = command.group(1).rsplit("/", 1)[-1].lower()
        if program in READ_COMMANDS:
            return "read"
        if program in WRITE_COMMANDS:
            return "write"
    named = re.search(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", action) \
        or re.search(r"Action\s*:\s*([A-Za-z_][\w.\-]*)", action)
    text = named.group(1) if named else action[:80]
    words = re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower())
    for word in words:
        if word in READ_VERBS:
            return "read"
        if word in WRITE_VERBS:
            return "write"
    return "other"


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------
def _make_slot(index, fmt, leaf, tool, via=(), anchor=None, prior=None):
    key = leaf.path[-1] if leaf.path else None
    if fmt == "reprlike" and isinstance(key, str):
        key = key.split(":", 1)[1]
    kind, words = classify_text(leaf.value, key)
    if kind == "free_text" and prior and _echoes(leaf.value, prior):
        kind = "echo"                         # the agent's own words, shown back to it by a display or log tool
    return {"index": index, "fmt": fmt, "path": list(leaf.path),
            "via": [{"fmt": v["fmt"], "path": list(v["path"])} for v in via],
            "key": key if isinstance(key, str) else None, "kind": kind, "free_text": kind == "free_text",
            "words": words, "chars": len(leaf.value), "offset": leaf.end if anchor is None else anchor,
            "inner_offset": leaf.end if via else 0, "tool_kind": tool}


def enumerate_slots(traj, probe=True):
    """Every string value inside the tool outputs -> (slots, rejected).

    With `probe`, every free-text slot is test-edited with a hostile string and dropped (into `rejected`) when the
    edit does not verify, so a remaining slot accepts any suffix text.
    """
    slots, rejected = [], []
    prior = _squash(traj.get("request") or "")
    for index, message in enumerate(traj["messages"]):
        if message["role"] == "agent":
            prior += _squash(_agent_text(message))
            continue
        if message["role"] != "observation":
            continue
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        action = next((m.get("action") for m in reversed(traj["messages"][:index]) if m["role"] == "agent"), None)
        kind = tool_kind(action)
        fmt = detect_format(text)
        if fmt == "text" and looks_structured(text):
            rejected.append({"index": index, "reason": "structured_unparsed"})
            continue
        for via, leaf_fmt, leaf, anchor in layered_leaves(text, fmt):
            slots.append(_make_slot(index, leaf_fmt, leaf, kind, via, anchor, prior))
    if not probe:
        return slots, rejected
    good = []
    for slot in slots:
        if slot["free_text"]:
            try:
                apply_slot(traj, slot, PROBE)
            except SlotError as exc:
                rejected.append({"index": slot["index"], "path": slot["path"], "reason": str(exc)})
                continue
        good.append(slot)
    return good, rejected


def select_slot(traj, slots=None):
    """The place for the suffix in `traj`, or None when there is none.

    Among the free-text strings of the tool outputs, the last one that follows a read-style tool call (third-party
    content), else the last one that follows a call of unknown kind. Tool outputs after a write-style call (create,
    send, delete ...) acknowledge the agent's own action and are never used.
    """
    slots = enumerate_slots(traj)[0] if slots is None else slots
    pool = [s for s in slots if s["free_text"] and s["tool_kind"] != "write"]
    pool = [s for s in pool if s["tool_kind"] == "read"] or pool
    if not pool:
        return None
    return max(pool, key=lambda s: (s["index"], s["offset"], s["inner_offset"]))


# --------------------------------------------------------------------------
# Applying and verifying an edit
# --------------------------------------------------------------------------
def _common_prefix(a, b):
    """Length of the longest common prefix, by bisection (slices compare in C)."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        mid = (low + high + 1) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def _single_insertion(old, new):
    """The text inserted into `old` to make `new`, or None if `new` is not `old` plus one contiguous insertion."""
    if len(new) < len(old):
        return None
    prefix = _common_prefix(old, new)
    suffix = _common_prefix(old[prefix:][::-1], new[::-1])       # cannot overlap the prefix inside `old`
    if prefix + suffix != len(old):
        return None
    return new[prefix:len(new) - suffix]


def _listify(obj):
    if isinstance(obj, (list, tuple)):
        return [_listify(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _listify(value) for key, value in obj.items()}
    return obj


def _parsed(text, fmt):
    if fmt == "json":
        return json.loads(text)
    if fmt == "pylit":
        return _listify(ast.literal_eval(text.strip()))
    if fmt == "reprlike":          # every field's decoded value, plus the text between the values
        leaves = reprlike_leaves(text)
        pieces, last = [], 0
        for leaf in leaves:
            pieces.append(text[last:leaf.start])
            last = leaf.end
        pieces.append(text[last:])
        return {**{leaf.path[0]: leaf.value for leaf in leaves}, "\x00skeleton": pieces}
    return text


def _get(obj, path):
    for step in path:
        obj = obj[step]
    return obj


def _set(obj, path, value):
    if not path:
        return value
    _get(obj, path[:-1])[path[-1]] = value
    return obj


def _layer_texts(text, layers):
    """The text of each layer: the container itself, then the string value that holds each inner container."""
    texts = [text]
    for spec in layers[:-1]:
        texts.append(_get(_parsed(texts[-1], spec["fmt"]), spec["path"]))
    return texts


def _expanded(text, layers):
    """The parsed structure with each carrier string replaced by the structure it holds."""
    spec = layers[0]
    obj = _parsed(text, spec["fmt"])
    if len(layers) == 1:
        return obj
    return _set(obj, spec["path"], _expanded(_get(obj, spec["path"]), layers[1:]))


def verify_edit(old, new, slot, suffix):
    """Problems with an edit (empty list = clean). Checks everything the attack claims about it."""
    problems = []
    try:
        layers = slot_layers(slot)
        old_text, new_text = old["messages"][slot["index"]]["content"], new["messages"][slot["index"]]["content"]
        # nothing outside the one container changed
        if set(old) != set(new):
            problems.append(f"top-level keys changed: {sorted(set(old) ^ set(new))}")
        for key in old:
            if key == "messages":
                if len(old[key]) != len(new[key]):
                    problems.append(f"{key} length changed")
                    continue
                for i, (a, b) in enumerate(zip(old[key], new[key])):
                    if i == slot["index"]:
                        if {k: v for k, v in a.items() if k != "content"} != {k: v for k, v in b.items() if k != "content"}:
                            problems.append("another field of the edited item changed")
                    elif a != b:
                        problems.append(f"{key}[{i}] changed")
            elif old[key] != new.get(key):
                problems.append(f"{key} changed")
        # every layer grew by exactly one insertion and is still the same format
        old_texts, new_texts = _layer_texts(old_text, layers), _layer_texts(new_text, layers)
        for number, (before, after, spec) in enumerate(zip(old_texts, new_texts, layers)):
            if _single_insertion(before, after) is None:
                problems.append(f"layer {number} is not the old text plus one insertion")
            if detect_format(after) != spec["fmt"]:
                problems.append(f"layer {number} format changed from {spec['fmt']} to {detect_format(after)}")
        # the whole structure parses to the old one with exactly one string grown by exactly the intended text
        leaf = _find_leaf(old_texts[-1], layers[-1]["fmt"], layers[-1]["path"])
        wanted = leaf.value + _separator(leaf.value) + suffix
        full_path = [step for spec in layers for step in spec["path"]]
        expected = _set(_expanded(old_text, layers), full_path, wanted)
        if repr(expected) != repr(_expanded(new_text, layers)):
            problems.append("the parsed structure is not the old one with exactly the target string grown by the suffix")
        # the judge's input differs by exactly that insertion. The same new string can come from inserting the
        # text at two neighbouring places, where the two readings are rotations of each other (the text after
        # the container may start like the inserted text), so rotations count as the same insertion.
        inserted = _single_insertion(old_text, new_text)
        old_state, new_state = render_state(old)["trajectory"], render_state(new)["trajectory"]
        shown = _single_insertion(old_state, new_state)
        if inserted is not None and (shown is None or len(shown) != len(inserted) or shown not in inserted + inserted):
            problems.append("rendered judge input does not differ by exactly the inserted text")
    except (SlotError, ValueError, KeyError, IndexError, TypeError, SyntaxError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    return problems


def apply_slot(traj, slot, suffix, verify=True):
    """A copy of `traj` with `suffix` inserted at `slot`. The original is never modified.

    The copy shares everything it did not change (other messages, tools, meta) with the original: treat both as
    immutable.

    Raises SlotError if the edit cannot be made cleanly. An empty suffix returns `traj` unchanged.
    """
    if not suffix:
        return traj
    index, items = slot["index"], traj["messages"]
    if not 0 <= index < len(items) or not isinstance(items[index].get("content"), str):
        raise SlotError(f"slot points at a missing or non-text field: messages[{index}].content")
    if items[index]["role"] != "observation":
        raise SlotError(f"only observations can be edited, not {items[index]['role']} messages")
    new_text, _, _ = edit_layers(items[index]["content"], slot_layers(slot), suffix)
    changed = list(items)
    changed[index] = {**items[index], "content": new_text}
    edited = {**traj, "messages": changed}
    if verify:
        problems = verify_edit(traj, edited, slot, suffix)
        if problems:
            raise SlotError("; ".join(problems))
    return edited
