"""Regenerate the paper's tables from the released result files, as decimals. No API calls.

    python -m scripts.paper.tables

Prints the main table (Table 1) and the appendix tables, and writes the three generated LaTeX table bodies
(token ablation, per-question re-evaluation, Choice-answer search) to the output directory (TABLE_DIR, default
tables/). Every printed number can be checked against sections/04_experiments.tex and the appendix.
"""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = Path(os.environ.get("TABLE_DIR", REPO / "tables"))
# key, short name (paper order of the nine classification datasets).
DATASETS = [("trec", "TREC"), ("ag_news", "AG"), ("dbpedia_14", "DBP"), ("yahoo_answers", "Yahoo"),
            ("patent", "Patent"), ("arxiv_abstracts", "arXivA"), ("20newsgroups", "20NG"),
            ("lexglue_ecthr", "ECtHR"), ("arxiv_fulltext", "arXivF")]
BENCHES = [("rjudge", "R-Judge"), ("atbench", "ATBench")]
QUESTIONS = [("plain_noul", "Noul"), ("boundary_noul", "b-Noul"), ("plain_choice", "Choice"), ("boundary_choice", "b-Choice")]
ABLATION_VARIANTS = [
    ("original", "the universal suffix, re-evaluated"),
    ("label_swapped", "label word swapped (\\sfx{no}$\\to$\\sfx{yes}, \\sfx{false}$\\to$\\sfx{true})"),
    ("label_deleted", "label word deleted"), ("question_deleted", "\\sfx{question} deleted"),
    ("shuffled", "tokens shuffled"), ("label_only", "label word alone"),
    ("seed_text", "the seed, \\sfx{question: is 'no' accurate}"), ("none", "no suffix")]


def load(path):
    return json.loads((RESULTS / path).read_text())


def ratio(flipped, evaluable):
    return f"{flipped / evaluable:.2f}" if evaluable else "--"


def main_table():
    """Table 1: share of correct, confident decisions that each method flips, on all 12 datasets."""
    columns = ["Email"] + [s for _, s in DATASETS] + [s for _, s in BENCHES]
    rows = {"Common injections": [], "Seed text, no search": [], "Per-document search": [], "Universal suffix": []}

    # Common injections.
    rows["Common injections"].append("0.00")   # emails: 0 strict flips in 10,240 paired tests (RQ1)
    for key, _ in DATASETS:
        inj = load(f"common_injections/datasets/{key}.json")
        flipped = sum(v["flipped"] for v in inj.values())
        evaluable = sum(v["evaluable"] for v in inj.values())
        rows["Common injections"].append(ratio(flipped, evaluable))
    for key, _ in BENCHES:
        inj = load(f"common_injections/trajectories/{key}.json")
        flipped = sum(v["flipped"] for v in inj.values())
        evaluable = sum(v["evaluable"] for v in inj.values())
        rows["Common injections"].append(ratio(flipped, evaluable))

    # Seed text, no search.
    seed = load("universal/emails/seed_text.json")
    rows["Seed text, no search"].append(ratio(seed["flipped"], seed["evaluable"]))
    ablation = load("ablation/summary.json")
    for key, _ in DATASETS:
        variant = ablation[key]["variants"]["seed_text"]
        rows["Seed text, no search"].append(ratio(variant["flips"]["plain_noul"], variant["n_docs"]))
    for key, _ in BENCHES:
        seed = load(f"universal/trajectories/{key}/controls_extra.json")["unevolved_seed"]
        rows["Seed text, no search"].append(ratio(seed["flipped"], seed["evaluable"]))

    # Per-document search.
    emails = sum(1 for p in sorted((RESULTS / "per_document" / "emails").glob("campaign-*/*/result.json"))
                 if json.loads(p.read_text()).get("success"))
    rows["Per-document search"].append(f"{emails / 28:.2f}")
    for key, _ in DATASETS:
        s = load(f"per_document/datasets/{key}/asr/asr_summary.json")
        rows["Per-document search"].append(ratio(s["search_success"], s["targets"]))
    for key, _ in BENCHES:
        s = load(f"per_document/trajectories/{key}/asr/asr_summary.json")
        rows["Per-document search"].append(ratio(s["search_success"], s["targets"]))

    # Universal suffix.
    u = load("universal/emails/universal.json")
    rows["Universal suffix"].append(ratio(u["heldout_flipped"], u["heldout_evaluable"]))
    for key, _ in DATASETS:
        u = load(f"universal/datasets/{key}/universal.json")
        rows["Universal suffix"].append(ratio(u["heldout_flipped"], u["heldout_evaluable"]))
    for key, _ in BENCHES:
        u = load(f"universal/trajectories/{key}/universal.json")
        rows["Universal suffix"].append(ratio(u["heldout_flipped"], u["heldout_evaluable"]))

    print("Table 1: attack success rate on the 12 datasets")
    print("  " + " ".join(f"{c:>7s}" for c in ["Method"] + columns))
    for name, cells in rows.items():
        print(f"  {name:21s} " + " ".join(f"{c:>7s}" for c in cells))
    return rows


def ablation_table():
    """Token ablation per dataset (plain Noul), as decimals, plus the Choice-answer re-evaluation bodies."""
    summary = load("ablation/summary.json")
    lines = []
    for key, label in ABLATION_VARIANTS:
        cells, flips, docs = [], 0, 0
        for ds, _ in DATASETS:
            variant = summary.get(ds, {}).get("variants", {}).get(key)
            if variant is None or "skipped" in variant:
                cells.append("--")
                continue
            n, f = variant["n_docs"], variant["flips"]["plain_noul"]
            cells.append(f"{f / n:.2f}")
            flips += f
            docs += n
        mean = f"{flips / docs:.2f}" if docs else "--"
        lines.append(f"    {label} & " + " & ".join(cells) + f" & {mean} \\\\")
    (OUT / "tab_ablation_body.tex").write_text("\n".join(lines) + "\n")

    heads = []
    for ds, short in DATASETS:
        variant = summary[ds]["variants"]["original"]
        n = variant["n_docs"]
        cells = " & ".join(f"{variant['flips'][q] / n:.2f}" for q, _ in QUESTIONS)
        heads.append(f"    {short} & {cells} & {variant['mean_p_no']:.2f} \\\\")
    (OUT / "tab_heads_body.tex").write_text("\n".join(heads) + "\n")

    print("\nToken ablation (plain Noul), per dataset and overall:")
    for line in lines:
        print("  " + line.strip().replace(" \\\\", ""))
    return lines, heads


def choice_table():
    """Universal suffixes searched against the Choice answer (three datasets)."""
    rows = []
    for ds, short in DATASETS:
        path = RESULTS / "choice_search" / ds / "universal.json"
        if not path.exists():
            continue
        u = json.loads(path.read_text())
        flips = u["heldout_flips_by_question"]
        n = u["heldout_evaluable"]
        rows.append(f"    {short} & {'yes' if u['search_success'] else 'no'} & {u['candidates_run']:,} & "
                    f"{flips['plain_choice'] / n:.2f} & {flips['boundary_choice'] / n:.2f} & "
                    f"{flips['plain_noul'] / n:.2f} & {flips['boundary_noul'] / n:.2f} \\\\")
    (OUT / "tab_choice_body.tex").write_text("\n".join(rows) + "\n")
    print("\nChoice-answer search (Thr, Cand, Choice, b-Choice, Noul, b-Noul):")
    for row in rows:
        print("  " + row.strip().replace(" \\\\", ""))
    return rows


def universal_table():
    """Universal searches of the nine datasets: whether the training mean crossed, candidates and held-out share."""
    print("\nUniversal searches (crossed, candidates, held-out share):")
    for key, short in DATASETS:
        u = load(f"universal/datasets/{key}/universal.json")
        print(f"  {short:7s} {'yes' if u['search_success'] else 'no':>3s}  {u['candidates_run']:>5,}  "
              f"{u['heldout_flipped'] / u['heldout_evaluable']:.2f}")


def judge_table():
    """Jev as a trajectory judge: per-target and universal flip rates and score drops."""
    print("\nTrajectory judge (per-target re-test and universal held-out):")
    retest = load("per_document/trajectories/retest.json")
    for key, short in BENCHES:
        entries = retest["results"][key]
        best = [e["texts"]["best_found"] for e in entries]
        per_target = sum(t["flipped_expected"] for t in best) / len(best)
        drop = sum(t["drop"] for t in best) / len(best)
        u = load(f"universal/trajectories/{key}/universal.json")
        print(f"  {short:7s} per-target flip {per_target:.2f} drop {drop:.2f}; "
              f"universal flip {u['heldout_flipped'] / u['heldout_evaluable']:.2f} drop stored in heldout records")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_table()
    ablation_table()
    choice_table()
    universal_table()
    judge_table()
    print(f"\nwrote the generated table bodies to {OUT}")


if __name__ == "__main__":
    main()
