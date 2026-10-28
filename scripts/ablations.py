"""Appendix analyses: the token ablation of the universal suffixes, and the Choice-answer search.

The token ablation re-evaluates each universal suffix and seven variants of it (label word swapped, deleted or kept
alone; the word "question" deleted; the tokens shuffled; the seed; no suffix) on the held-out documents under all four
questions. The Choice-answer search runs the same attack with the plain Choice answer as the fitness.

    python -m scripts.ablations token --universal-root runs/universal --out runs/ablation
    python -m scripts.ablations choice --dataset trec --universal-root runs/universal --out runs/choice
    python -m scripts.ablations token --universal-root runs/universal --dry-run
"""

import argparse
import json
from pathlib import Path
import random
import re

from flipjev import datasets as D
from flipjev.api import api_key
from flipjev.evaluation import evaluate_suffix, majority_flips, make_universal_evaluator
from flipjev.search import MAX_LENGTH, Search, confirmed_text
from flipjev.study import QueryLog, write_json
from flipjev.text import encoded_length, render, tokenize

QUESTIONS = ["plain_noul", "boundary_noul", "plain_choice", "boundary_choice"]
SEED_TEXT = "question: is 'no' accurate"
LABEL_PATTERNS = [(r"'no'", "'yes'"), (r"\bNO\b", "YES"), (r"\bno\b", "yes"), (r"\bfalse\b", "true"), (r"\bFalse\b", "True")]


def _squeeze(text):
    return re.sub(r"\s+", " ", text).strip()


def variants(suffix, rng):
    """The suffix and seven variants of it; a variant that does not change the suffix is dropped."""
    out = {"original": suffix}
    swapped = suffix
    for pattern, replacement in LABEL_PATTERNS:
        swapped = re.sub(pattern, replacement, swapped)
    out["label_swapped"] = swapped
    deleted = suffix
    for pattern, _ in LABEL_PATTERNS:
        deleted = re.sub(pattern, "", deleted)
    out["label_deleted"] = _squeeze(deleted)
    if re.search(r"\bquestions?\b", suffix, flags=re.I):
        out["question_deleted"] = _squeeze(re.sub(r"\bquestions?\b", "", suffix, flags=re.I))
    for pattern, _ in LABEL_PATTERNS:
        match = re.search(pattern, suffix)
        if match:
            out["label_only"] = match.group(0)
            break
    tokens = tokenize(suffix)
    for _ in range(50):
        shuffled = tokens[:]
        rng.shuffle(shuffled)
        text = render(shuffled)
        if text != suffix and encoded_length(text) <= MAX_LENGTH:
            out["shuffled"] = text
            break
    out["seed_text"] = SEED_TEXT
    out["none"] = ""
    return {name: text for name, text in out.items() if name == "original" or text != suffix}


def token_ablation(key, universal_root, out, repeats, workers, datasets):
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in datasets:
        spec = D.task_for(ds)
        by_id = {c["case_id"]: c for c in D.load_cases(ds)}
        selection = json.loads((universal_root / ds / "selection.json").read_text())
        test = [by_id[i] for i in selection["test_ids"] if i in by_id]
        suffix = json.loads((universal_root / ds / "universal.json").read_text())["suffix"]
        summary[ds] = {"suffix": suffix, "variants": {}}
        for name, text in variants(suffix, random.Random(20261005)).items():
            if text and (not spec.allowed(text) or encoded_length(text) > MAX_LENGTH):
                summary[ds]["variants"][name] = {"text": text, "skipped": "forbidden or too long"}
                continue
            result = evaluate_suffix(spec, test, text, key, repeats=repeats, workers=workers)
            by_q = {q: majority_flips(result["records"], q, repeats) for q in QUESTIONS}
            ok = [r for r in result["records"] if r["status"] == "ok"]
            mean_p_no = sum(r["scores"]["plain_noul"] for r in ok) / len(ok) if ok else None
            summary[ds]["variants"][name] = {"text": text, "n_docs": result["n_evaluable"], "flips": by_q,
                                             "mean_p_no": mean_p_no}
            print(f"[{ds}] {name:16s} noul {by_q['plain_noul']}/{result['n_evaluable']}")
    write_json(out / "summary.json", summary)
    return summary


def choice_search(key, dataset, universal_root, out, budget, repeats, workers):
    out = out / dataset
    out.mkdir(parents=True, exist_ok=True)
    spec = D.task_for(dataset, primary="plain_choice")
    by_id = {c["case_id"]: c for c in D.load_cases(dataset)}
    selection = json.loads((universal_root / dataset / "selection.json").read_text())
    train = [by_id[i] for i in selection["train_ids"] if i in by_id]
    test = [by_id[i] for i in selection["test_ids"] if i in by_id]
    log = QueryLog(out / "queries")
    evaluate = make_universal_evaluator(spec, train, key, workers=workers)
    search = Search(evaluate, spec.seeds["question"], random.Random(0), spec.vocab, budget=budget, elite=16,
                    children=64, workers=1, confirm=3, primary=spec.primary, allowed=spec.allowed,
                    max_length=spec.max_length, on_query=log.for_search("choice"))
    result = search.run()
    suffix = confirmed_text(result, spec.primary) if result["success"] else result["best"]["text"]
    held = evaluate_suffix(spec, test, suffix, key, repeats=repeats, workers=workers)
    out_json = {"dataset": dataset, "primary": "plain_choice", "search_success": result["success"], "suffix": suffix,
                "tokens": encoded_length(suffix), "candidates_run": result["queries"],
                "heldout_evaluable": held["n_evaluable"],
                "heldout_flips_by_question": {q: majority_flips(held["records"], q, repeats) for q in QUESTIONS}}
    write_json(out / "universal.json", out_json)
    print(json.dumps({k: out_json[k] for k in ("search_success", "candidates_run", "heldout_flips_by_question")}, indent=2))
    return out_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["token", "choice"])
    parser.add_argument("--dataset", choices=sorted(D.REGISTRY), help="one dataset (choice search)")
    parser.add_argument("--datasets", nargs="+", default=D.TOPIC_DATASETS, help="datasets for the token ablation")
    parser.add_argument("--universal-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=5000, help="candidate budget for the Choice-answer search")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"analysis": args.analysis, "dataset": args.dataset, "datasets": args.datasets}, indent=2))
        return
    if not args.out:
        parser.error("--out is required")
    key = api_key()
    if args.analysis == "token":
        token_ablation(key, args.universal_root, args.out, args.repeats, args.workers, args.datasets)
    else:
        if not args.dataset:
            parser.error("choice search needs --dataset")
        choice_search(key, args.dataset, args.universal_root, args.out, args.budget, args.repeats, args.workers)


if __name__ == "__main__":
    main()
