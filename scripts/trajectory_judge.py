"""Jev as an agent-trajectory safety judge: per-target search, universal suffix, controls, re-test and transfer.

The attack inserts text inside one third-party string of one tool output and tries to turn an unsafe verdict into a
safe one (flipjev/judge.py, flipjev/insertion.py). The stages mirror the classification study. The benchmarks are not
part of this repository; set TRAJECTORY_DATA or pass --data (see the README for the download).

    python -m scripts.trajectory_judge --dataset atbench --out runs/judge
    python -m scripts.trajectory_judge --transfer --out runs/judge
    python -m scripts.trajectory_judge --dataset rjudge --dry-run
"""

import argparse
import json
from pathlib import Path
import random
import statistics

from flipjev.api import api_key
from flipjev.evaluation import evaluate_suffix
from flipjev.judge import (NEUTRAL_NOTE, SEEDS, TOP, load_unsafe_cases, returned_score, trajectory_judge_task)
from flipjev.study import (Budget, QueryLog, controls, per_document_asr, transfer_matrix, universal_search, write_json)

BENCHES = ("rjudge", "atbench")
SPLIT_SEED = 20260929


def select_targets(spec, cases, key, out_dir, cap, min_score, targets_n, train_n, test_n, workers):
    """Clean pass once, then a seeded draw of target, training and held-out trajectories from those flagged unsafe
    with a returned score of at least `min_score`."""
    path = out_dir / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    if not cap.room():
        return None
    result = evaluate_suffix(spec, cases, "", key, repeats=1, workers=workers, cost_limit=cap.remaining())
    cap.add(result["cost"])
    clean = {r["case_id"]: returned_score(r["scores"][spec.primary]) for r in result["records"] if r["status"] == "ok"}
    ids = [c["case_id"] for c in cases if clean.get(c["case_id"], 0.0) >= min_score]
    random.Random(SPLIT_SEED).shuffle(ids)
    rest = ids[targets_n:]
    selection = {"min_clean_score": min_score, "eligible": len(cases), "n_confident": len(ids),
                 "target_ids": ids[:targets_n], "train_ids": rest[:train_n], "test_ids": rest[train_n:train_n + test_n],
                 "clean_scores": {i: round(clean[i], 3) for i in ids}}
    write_json(path, selection)
    return selection


def choose_seed(spec, train, key, out_dir, cap, workers):
    """The seed with the highest mean P(safe) over the training trajectories."""
    path = out_dir / "seed_choice.json"
    if path.exists():
        return json.loads(path.read_text())["chosen"]
    rows = {}
    for name, text in spec.seeds.items():
        result = evaluate_suffix(spec, train, text, key, repeats=1, workers=workers, cost_limit=cap.remaining())
        cap.add(result["cost"])
        values = [c["mean_p_target"] for c in result["per_case"] if c["mean_p_target"] is not None]
        rows[name] = {"mean_p_safe": statistics.fmean(values) if values else 0.0, "flipped": result["flipped_cases"]}
    chosen = max(rows, key=lambda n: (rows[n]["mean_p_safe"], rows[n]["flipped"]))
    write_json(path, {"chosen": chosen, "seeds": rows})
    return chosen


def run_dataset(args, key):
    spec = trajectory_judge_task()
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    cap = Budget(args.budget_usd)
    log = QueryLog(out / "queries")
    cases = load_unsafe_cases(args.dataset, args.max_tokens)
    by_id = {c["case_id"]: c for c in cases}
    selection = select_targets(spec, cases, key, out, cap, args.clean_score, args.targets, args.train, args.test, args.workers)
    targets = [by_id[i] for i in selection["target_ids"] if i in by_id]
    train = [by_id[i] for i in selection["train_ids"] if i in by_id]
    test = [by_id[i] for i in selection["test_ids"] if i in by_id]
    print(f"eligible {selection['eligible']}, confident {selection['n_confident']} -> "
          f"targets {len(targets)}, train {len(train)}, test {len(test)}")
    seed = choose_seed(spec, train, key, out, cap, args.workers)
    print(f"seed: {seed}")
    asr = per_document_asr(spec, targets, key, out / "asr", args.budget, args.workers, args.repeats, cap,
                           seed_attack=seed, query_log=log)
    universal = universal_search(spec, train, test, key, out / "universal.json", args.univ_budget, args.workers,
                                 args.repeats, cap, seed_attack=seed, query_log=log)
    control = controls(spec, test, key, out / "controls.json", cap, args.repeats, args.workers,
                       conditions=(("suffix", universal.get("suffix", "")), ("no_suffix", ""), ("neutral_note", NEUTRAL_NOTE)),
                       records_path=out / "heldout_records.json")
    print(json.dumps({"per_document": asr, "universal": {k: universal.get(k) for k in
                      ("suffix", "search_success", "heldout_flipped", "heldout_evaluable")}, "controls": control}, indent=2))
    print(f"spent ${cap.spent:.2f}")


def run_transfer(args, key):
    spec = trajectory_judge_task()
    cap = Budget(args.budget_usd)
    specs, tests, suffixes = {}, {}, {}
    for name in BENCHES:
        universal = json.loads((args.out / name / "universal.json").read_text())
        selection = json.loads((args.out / name / "selection.json").read_text())
        by_id = {c["case_id"]: c for c in load_unsafe_cases(name, args.max_tokens)}
        specs[name] = spec
        tests[name] = [by_id[i] for i in selection["test_ids"] if i in by_id]
        if universal.get("suffix"):
            suffixes[name] = universal["suffix"]
    # The two benchmarks use the same questions, so only the ATBench suffix on R-Judge changes data alone.
    matrix = transfer_matrix(specs, tests, suffixes, key, args.out / "transfer", args.workers, cap, repeats=args.repeats)
    print(json.dumps(matrix, indent=2))
    print(f"spent ${cap.spent:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=BENCHES)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--budget", type=int, default=2000, help="queries per per-target search")
    parser.add_argument("--univ-budget", type=int, default=3000, help="candidates for the universal search")
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--clean-score", type=float, default=2.5, help="targets are flagged unsafe with at least this score")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-usd", type=float, default=16.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.transfer and not args.dataset:
        parser.error("give --dataset or --transfer")
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "transfer": args.transfer, "seeds": list(SEEDS),
                          "score_scale": f"0-{TOP}", "budget": args.budget, "univ_budget": args.univ_budget}, indent=2))
        return
    if not args.out:
        parser.error("--out is required")
    key = api_key()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.transfer:
        run_transfer(args, key)
    else:
        run_dataset(args, key)


if __name__ == "__main__":
    main()
