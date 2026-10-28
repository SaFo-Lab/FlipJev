"""The nine classification datasets and IMDB: binary questions about native classes of public datasets.

Each dataset becomes one yes/no question about one of its classes (the positive class is the attack target), with a
list of forbidden words that name that class, so a suffix cannot add or remove a mention of the subject. A builder
draws 150 positive and 150 negative documents with a fixed seed from the training split, cuts each document at a fixed
number of characters, drops documents with fewer than 40 characters and writes cases.jsonl. The Hugging Face datasets
are pinned to the revisions that were used, so the rebuilt files are byte-identical (data/datasets/checksums.json).
The arXiv abstracts came from the live arXiv API and are shipped with the repository instead.
"""

import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from flipjev.tasks import text_classification_task

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "datasets"
SHUFFLE_SEED = 20260929
MIN_CHARS = 40
TOPIC_DATASETS = ["trec", "ag_news", "dbpedia_14", "yahoo_answers", "patent", "arxiv_abstracts", "20newsgroups",
                  "lexglue_ecthr", "arxiv_fulltext"]
SENTIMENT_DATASETS = ["imdb"]   # appended text can change the sentiment, so IMDB is reported separately


def _clean(text):
    text = re.sub(r"\s+\n", "\n", (text or "").strip())
    return re.sub(r"\n{3,}", "\n\n", text)


def hf_rows(name, config, revision, get_text, is_positive, limit, scan=8000):
    """(text, is_positive) for the first `limit` rows of a seeded shuffle of the training split.

    The whole split is shuffled in memory because several mirrors store their rows sorted by label.
    """
    from datasets import load_dataset
    ds = load_dataset(name, config, split="train", revision=revision)
    ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(min(len(ds), scan)))
    for n, row in enumerate(ds, 1):
        text = _clean(get_text(row))
        if text:
            yield text, bool(is_positive(row))
        if n >= limit:
            break


def newsgroups_rows(topic, negatives, limit):
    from sklearn.datasets import fetch_20newsgroups
    bunch = fetch_20newsgroups(subset="train", categories=[topic] + negatives,
                               remove=("headers", "quotes", "footers"), shuffle=True, random_state=SHUFFLE_SEED)
    index = bunch.target_names.index(topic)
    for n, (text, target) in enumerate(zip(bunch.data, bunch.target), 1):
        text = _clean(text)
        if text:
            yield text, target == index
        if n >= limit:
            break


def arxiv_api_rows(positive_category, negative_categories, limit):
    """The 120 most recent abstracts of each category from the public arXiv API (a new call returns newer papers)."""
    atom = "http://www.w3.org/2005/Atom"
    for category, is_positive in [(positive_category, True)] + [(c, False) for c in negative_categories]:
        url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(f"cat:{category}")
               + "&start=0&max_results=120&sortBy=submittedDate&sortOrder=descending")
        with urllib.request.urlopen(url, timeout=40) as response:
            root = ET.fromstring(response.read())
        for entry in root.findall(f"{{{atom}}}entry"):
            text = _clean(" ".join((entry.findtext(f"{{{atom}}}summary") or "").split()))
            if text:
                yield text, is_positive
        time.sleep(3)


# ccdv/arxiv-classification label names; the cs.* labels form the positive class.
_ARXIV_LABELS = ["math.AC", "cs.CV", "cs.AI", "cs.SY", "math.GR", "cs.CE", "cs.PL", "cs.IT", "cs.DS", "cs.NE", "math.ST"]
_ARXIV_CS = {i for i, n in enumerate(_ARXIV_LABELS) if n.startswith("cs.")}
SPORTS = [r"sport", r"\bgame[sd]?\b", r"\bteam", r"player", r"league", r"\bmatch", r"championship",
          r"coach", r"tournament", r"\bathlet"]
COMPUTING = [r"comput", r"algorithm", r"software", r"machine\s*learn", r"neural", r"\bdata\b",
             r"\bAI\b", r"network", r"program", r"\bcode\b"]

REGISTRY = {
    "trec": dict(  # coarse label "human beings"
        question="Is this question asking about a person or people?",
        yes="The question asks about a person or people.", no="The question does not ask about a person or people.",
        forbidden=[r"\bwho\b", r"person", r"people", r"\bname[sd]?\b", r"someone", r"human"],
        source="SetFit/TREC-QC", revision="0a34640b3cac3affef6f0abdfb81e3d0ecffcf92",
        loader=lambda limit: hf_rows("SetFit/TREC-QC", None, REGISTRY["trec"]["revision"],
                                     lambda r: r["text"], lambda r: r.get("label_coarse_text") == "human beings", limit),
        max_chars=400),
    "ag_news": dict(  # label 1 = Sports
        question="Is this news text about sports?",
        yes="The text is about sports.", no="The text is not about sports.",
        forbidden=SPORTS + [r"\bwon\b", r"season"],
        source="fancyzhx/ag_news", revision="eb185aade064a813bc0b7f42de02595523103ca4",
        loader=lambda limit: hf_rows("fancyzhx/ag_news", None, REGISTRY["ag_news"]["revision"],
                                     lambda r: r["text"], lambda r: r["label"] == 1, limit),
        max_chars=2000),
    "dbpedia_14": dict(  # label 0 = Company
        question="Is this text about a company or business organization?",
        yes="The text is about a company or business.", no="The text is not about a company or business.",
        forbidden=[r"compan(y|ies)", r"\bfirm", r"business", r"corporat", r"enterprise", r"\binc\b",
                   r"\bltd\b", r"manufactur", r"founded", r"headquarter"],
        source="fancyzhx/dbpedia_14", revision="9abd46cf7fc8b4c64290f26993c540b92aa145ac",
        loader=lambda limit: hf_rows("fancyzhx/dbpedia_14", None, REGISTRY["dbpedia_14"]["revision"],
                                     lambda r: r["content"], lambda r: r["label"] == 0, limit),
        max_chars=2000),
    "yahoo_answers": dict(  # topic 5 = Sports
        question="Is this text about sports?",
        yes="The text is about sports.", no="The text is not about sports.",
        forbidden=SPORTS,
        source="community-datasets/yahoo_answers_topics", revision="6652a1e7c94f7260a0bfd0c9092dd48e2d536ea1",
        loader=lambda limit: hf_rows("community-datasets/yahoo_answers_topics", None, REGISTRY["yahoo_answers"]["revision"],
                                     lambda r: " ".join([r.get("question_title", ""), r.get("question_content", ""),
                                                         r.get("best_answer", "")]),
                                     lambda r: r["topic"] == 5, limit),
        max_chars=2500),
    "20newsgroups": dict(  # sci.space against five other groups
        question="Is the main topic of the text outer space or astronomy?",
        yes="The main topic is outer space or astronomy.", no="The main topic is not outer space or astronomy.",
        forbidden=[r"space", r"spacecraft", r"orbit", r"nasa", r"astronom", r"planet", r"satellite", r"\blaunch",
                   r"rocket", r"shuttle", r"\bmoon", r"lunar", r"\bmars\b", r"solar", r"galax", r"telescope"],
        source="scikit-learn fetch_20newsgroups (train subset; headers, quotes and footers removed)", revision=None,
        loader=lambda limit: newsgroups_rows("sci.space", ["rec.autos", "comp.graphics", "talk.politics.mideast",
                                                           "rec.sport.hockey", "sci.med"], limit),
        max_chars=4000),
    "imdb": dict(  # label 1 = positive review; the attack pushes toward negative
        question="Is the sentiment of this movie review positive?",
        yes="The review expresses a positive sentiment.", no="The review expresses a negative sentiment.",
        forbidden=[],
        source="stanfordnlp/imdb", revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        loader=lambda limit: hf_rows("stanfordnlp/imdb", None, REGISTRY["imdb"]["revision"],
                                     lambda r: r["text"], lambda r: r["label"] == 1, limit),
        max_chars=4000),
    "patent": dict(  # abstract config, label 2 = Chemistry; Metallurgy
        question="Is this patent abstract about chemistry or metallurgy?",
        yes="The patent is about chemistry or metallurgy.", no="The patent is not about chemistry or metallurgy.",
        forbidden=[r"chemi", r"metallurg", r"\bcompound", r"molecul", r"reaction", r"\bacid", r"polymer",
                   r"catalyst", r"\balloy", r"\bsolvent"],
        source="ccdv/patent-classification (abstract)", revision="17bd94cd6594bac1b82a697e811fb7cf34a7af35",
        loader=lambda limit: hf_rows("ccdv/patent-classification", "abstract", REGISTRY["patent"]["revision"],
                                     lambda r: r["text"], lambda r: r["label"] == 2, limit),
        max_chars=3000),
    "arxiv_abstracts": dict(  # cs.LG against math.AG and q-bio.NC (the negatives fill up before the other two)
        question="Is this paper abstract about computer science?",
        yes="The abstract is about computer science.", no="The abstract is not about computer science.",
        forbidden=COMPUTING,
        source="arXiv API (export.arxiv.org), most recent submissions", revision=None,
        loader=lambda limit: arxiv_api_rows("cs.LG", ["math.AG", "q-bio.NC", "econ.EM", "astro-ph.GA"], limit),
        max_chars=3000),
    "arxiv_fulltext": dict(  # cs.* against math.* full papers
        question="Is this scientific paper about computer science?",
        yes="The paper is about computer science.", no="The paper is not about computer science.",
        forbidden=COMPUTING,
        source="ccdv/arxiv-classification (no_ref)", revision="d4bf86db1cecd1f42e5ef24f1718d7c02c3250a7",
        loader=lambda limit: hf_rows("ccdv/arxiv-classification", "no_ref", REGISTRY["arxiv_fulltext"]["revision"],
                                     lambda r: r["text"], lambda r: r["label"] in _ARXIV_CS, limit),
        max_chars=12000),
    "lexglue_ecthr": dict(  # ecthr_a; label names are ["2", "3", "5", ...], so index 1 is Article 3
        question="Does this case concern Article 3 of the Convention (prohibition of torture and inhuman or degrading treatment)?",
        yes="The case concerns Article 3 (torture / inhuman or degrading treatment).",
        no="The case does not concern Article 3.",
        forbidden=[r"tortur", r"inhuman", r"degrad", r"ill[- ]?treat", r"article\s*3", r"cruel", r"\babuse"],
        source="coastalcph/lex_glue (ecthr_a)", revision="c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        loader=lambda limit: hf_rows("coastalcph/lex_glue", "ecthr_a", REGISTRY["lexglue_ecthr"]["revision"],
                                     lambda r: "\n".join(r["text"]) if isinstance(r["text"], list) else r["text"],
                                     lambda r: 1 in r["labels"], limit),
        max_chars=12000),
}


def build(name, out_dir=DATA, per_label=150, candidate_limit=3000):
    """Draw the documents of one dataset and write <out_dir>/<name>/cases.jsonl and cases.meta.json."""
    config = REGISTRY[name]
    kept = {"yes": [], "no": []}
    for text, is_positive in config["loader"](candidate_limit):
        label = "yes" if is_positive else "no"
        if len(kept[label]) >= per_label:
            if all(len(v) >= per_label for v in kept.values()):
                break
            continue
        if len(text) < MIN_CHARS:
            continue
        kept[label].append(text[:config["max_chars"]])
    cases = []
    for label in ("yes", "no"):
        for i, text in enumerate(kept[label], 1):
            base_id = f"{label}_{i:03d}"
            cases.append({"case_id": base_id + "/clean", "base_id": base_id, "label": label, "text": text,
                          "stratum": name, "kind": "clean", "variant": "clean", "family": "clean",
                          "target": None, "control_case_id": None})
    directory = Path(out_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cases.jsonl"
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    meta = {"dataset": name, "source": config["source"], "revision": config["revision"],
            "question": config["question"], "yes": config["yes"], "no": config["no"], "forbidden": config["forbidden"],
            "labels": {label: len(kept[label]) for label in ("yes", "no")}, "max_chars": config["max_chars"],
            "case_count": len(cases), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def task_for(name, primary="plain_noul"):
    """The TaskSpec of a dataset; `primary` is the question whose P(no) is the fitness."""
    config = REGISTRY[name]
    return text_classification_task(name=name, instructions=config["question"], yes_criteria=config["yes"],
                                    no_criteria=config["no"], target="no", forbidden=config["forbidden"],
                                    state_field="input_text", primary=primary)


def load_cases(name, data_dir=DATA):
    path = Path(data_dir) / name / "cases.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; build it with: python -m scripts.build_datasets --datasets {name}")
    # Split on "\n" only: str.splitlines() also splits on characters such as "\x85" inside a document.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
