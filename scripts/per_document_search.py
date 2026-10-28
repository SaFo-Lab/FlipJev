"""RQ2: search a suffix for each target, on the email task or on one classification dataset.

For each target (a correct, confident yes document), the attack runs one search of up to --budget queries and then
re-queries the confirmed suffix on its own target. On the email task the targets are the nine long and hard yes emails
of the paper; on a dataset they are a seeded sample of its confident yes documents.

    python -m scripts.per_document_search --task email --out runs/email_search
    python -m scripts.per_document_search --task dataset --dataset trec --out runs/trec_search
    python -m scripts.per_document_search --task email --dry-run        # show the plan; send nothing
"""

import argparse
import json
from pathlib import Path

from flipjev import datasets as D
from flipjev.api import api_key
from flipjev.emails import email_classification_task
from flipjev.long_emails import long_email_cases
from flipjev.study import Budget, QueryLog, per_document_asr, select_targets

# The paper's email targets: four short hard emails and five long natural emails.
EMAIL_TARGETS = ["hard_yes_05", "hard_yes_08", "hard_yes_14", "hard_yes_15",
                 "long_yes_01", "long_yes_02", "long_yes_03", "long_yes_04", "long_yes_05"]


def email_targets():
    from flipjev.email_sets import hard_email_cases
    by_id = {c["base_id"]: c for c in hard_email_cases() if c["kind"] == "clean"}
    by_id.update({c["base_id"]: c for c in long_email_cases()})
    return [by_id[base] for base in EMAIL_TARGETS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["email", "dataset"], required=True)
    parser.add_argument("--dataset", choices=sorted(D.REGISTRY), help="required when --task dataset")
    parser.add_argument("--seed-attack", default="question", help="the seed text the search starts from")
    parser.add_argument("--budget", type=int, default=2000, help="queries per target")
    parser.add_argument("--targets", type=int, default=20, help="number of targets (datasets only)")
    parser.add_argument("--repeats", type=int, default=10, help="fresh queries to re-test each confirmed suffix")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-usd", type=float, default=10.0, help="hard cap on the total spend")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and send no query")
    args = parser.parse_args()
    if args.task == "dataset" and not args.dataset:
        parser.error("--task dataset needs --dataset")

    if args.dry_run:
        plan = {"task": args.task, "dataset": args.dataset, "seed_attack": args.seed_attack, "budget": args.budget,
                "targets": len(EMAIL_TARGETS) if args.task == "email" else args.targets, "model": "typesafe/jev-1.13"}
        print(json.dumps(plan, indent=2))
        return
    if not args.out:
        parser.error("--out is required")

    if args.task == "dataset":
        spec = D.task_for(args.dataset)
        cases = D.load_cases(args.dataset)
    else:
        spec = email_classification_task()
        cases = None

    key = api_key()
    args.out.mkdir(parents=True, exist_ok=True)
    cap = Budget(args.budget_usd)
    log = QueryLog(args.out / "queries")
    if args.task == "email":
        targets = email_targets()
    else:
        selection = select_targets(spec, cases, key, args.out, cap, targets_n=args.targets,
                                   workers=args.workers, seed_attack=args.seed_attack)
        by_id = {c["case_id"]: c for c in cases}
        targets = [by_id[i] for i in selection["target_ids"] if i in by_id]
    summary = per_document_asr(spec, targets, key, args.out / "asr", args.budget, args.workers, args.repeats, cap,
                               seed_attack=args.seed_attack, query_log=log)
    print(json.dumps(summary, indent=2))
    print(f"spent ${cap.spent:.2f}")


if __name__ == "__main__":
    main()
