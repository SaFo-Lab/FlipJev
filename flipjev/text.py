"""Suffix text: word-level tokens for the mutation operators, and length in o200k_base BPE tokens."""

import functools
import re

import tiktoken

# Length is measured in BPE tokens, not in words, so a suffix cannot avoid the limit by dropping spaces.
LENGTH_UNIT = "o200k_base"
_ENCODING = tiktoken.get_encoding(LENGTH_UNIT)


@functools.lru_cache(maxsize=100_000)
def encoded_length(text):
    return len(_ENCODING.encode(text))


def document_length(text):
    """BPE length of a document or a rendered trajectory, which may contain special-token strings as plain text."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def tokenize(text):
    return re.findall(r"\n|[\w'.]+|[^\w\s]", text)


def render(tokens):
    """Join tokens with ordinary spacing: no space before closing punctuation or after an opening bracket."""
    text = ""
    for token in tokens:
        if token == "\n" or not text or text.endswith("\n") or re.fullmatch(r"[.,!?:;)\]}]", token) or text.endswith(("(", "[", "{")):
            text += token
        else:
            text += " " + token
    return text.strip()
