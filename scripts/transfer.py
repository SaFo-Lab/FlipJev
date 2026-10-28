"""RQ4: apply the universal suffix of each dataset to the held-out documents of every other dataset.

Each dataset asks its own question, so a suffix that moves from one dataset to another meets both new data and a new
task. A suffix that the target task forbids (it would name the subject there) is skipped. This reads the universal
suffixes from an earlier universal_search run for each dataset.

    python -m scripts.transfer --universal-root runs/universal --out runs/transfer
    python -m scripts.transfer --universal-root runs/universal --dry-run
"""

import argparse
import json
from pathlib import Path

from flipjev import datasets as D
from flipjev.api import api_key
from flipjev.study import Budget, transfer_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universal-root", type=Path, required=True,
                        help="directory with <dataset>/universal.json and <dataset>/selection.json per dataset")
    parser.add_argument("--datasets", nargs="+", default=D.TOPIC_DATASETS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--first", type=int, default=20, help="held-out documents of each target dataset")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({"datasets": args.datasets, "universal_root": str(args.universal_root), "first": args.first}, indent=2))
        return
    if not args.out:
        parser.error("--out is required")

    specs, tests, suffixes = {}, {}, {}
    for name in args.datasets:
        root = args.universal_root / name
        universal = json.loads((root / "universal.json").read_text())
        selection = json.loads((root / "selection.json").read_text())
        by_id = {c["case_id"]: c for c in D.load_cases(name)}
        pool = selection.get("test_ids", [])
        specs[name] = D.task_for(name)
        tests[name] = [by_id[i] for i in pool if i in by_id]
        if universal.get("suffix"):
            suffixes[name] = universal["suffix"]

    key = api_key()
    cap = Budget(args.budget_usd)
    matrix = transfer_matrix(specs, tests, suffixes, key, args.out, args.workers, cap, repeats=args.repeats, first=args.first)
    print(json.dumps(matrix, indent=2))
    print(f"spent ${cap.spent:.2f}")


if __name__ == "__main__":
    main()
