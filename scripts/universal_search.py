"""RQ3: search one universal suffix on a batch of training documents and evaluate it on held-out documents.

The attack scores each candidate by its mean P(no) over the training documents, so the same search produces one
suffix for many inputs. The suffix is then applied, with no further search, to the held-out documents, next to two
controls (no suffix, and a neutral note).

    python -m scripts.universal_search --task email --out runs/email_universal
    python -m scripts.universal_search --task dataset --dataset dbpedia_14 --out runs/dbpedia_universal
    python -m scripts.universal_search --task dataset --dataset trec --dry-run
"""

import argparse
import json
from pathlib import Path

from flipjev import datasets as D
from flipjev.api import api_key
from flipjev.emails import email_classification_task
from flipjev.study import (Budget, NEUTRAL_NOTE, QueryLog, controls, select_targets, universal_search)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["email", "dataset"], required=True)
    parser.add_argument("--dataset", choices=sorted(D.REGISTRY), help="required when --task dataset")
    parser.add_argument("--seed-attack", default="question")
    parser.add_argument("--budget", type=int, default=5000, help="candidate budget for the universal search")
    parser.add_argument("--train", type=int, default=10, help="training documents")
    parser.add_argument("--test", type=int, default=30, help="held-out documents")
    parser.add_argument("--repeats", type=int, default=10, help="fresh queries per held-out document")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-usd", type=float, default=30.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.task == "dataset" and not args.dataset:
        parser.error("--task dataset needs --dataset")

    if args.dry_run:
        print(json.dumps({"task": args.task, "dataset": args.dataset, "seed_attack": args.seed_attack,
                          "budget": args.budget, "train": args.train, "test": args.test}, indent=2))
        return
    if not args.out:
        parser.error("--out is required")

    if args.task == "dataset":
        spec = D.task_for(args.dataset)
        cases = D.load_cases(args.dataset)
    else:
        spec = email_classification_task()
        from flipjev.email_sets import hard_email_cases
        from flipjev.long_emails import long_email_cases
        by_id = {c["base_id"]: c for c in hard_email_cases() if c["kind"] == "clean"}
        by_id.update({c["base_id"]: c for c in long_email_cases()})
        cases = list(by_id.values())

    key = api_key()
    args.out.mkdir(parents=True, exist_ok=True)
    cap = Budget(args.budget_usd)
    log = QueryLog(args.out / "queries")
    selection = select_targets(spec, cases, key, args.out, cap, targets_n=args.train + args.test,
                               train_n=args.train, test_n=args.test, workers=args.workers, seed_attack=args.seed_attack)
    by_id = {c["case_id"]: c for c in cases}
    # On a fresh dataset, the training and held-out documents are the first train+test of the sampled pool.
    pool = selection["target_ids"] + selection["train_ids"] + selection["test_ids"]
    train = [by_id[i] for i in pool[:args.train]]
    test = [by_id[i] for i in pool[args.train:args.train + args.test]]
    result = universal_search(spec, train, test, key, args.out / "universal.json", args.budget, args.workers,
                              args.repeats, cap, seed_attack=args.seed_attack, query_log=log)
    control = controls(spec, test, key, args.out / "controls.json", cap, args.repeats, args.workers,
                       conditions=(("no_suffix", ""), ("neutral_note", NEUTRAL_NOTE)))
    print(json.dumps({"universal": {k: result.get(k) for k in ("suffix", "search_success", "candidates_run",
                                                               "heldout_flipped", "heldout_evaluable")},
                      "controls": control}, indent=2))
    print(f"spent ${cap.spent:.2f}")


if __name__ == "__main__":
    main()
