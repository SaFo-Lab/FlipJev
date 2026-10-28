"""Regenerate the paper's data figures from the released result files. No API calls.

    python -m scripts.paper.figures            # all figures into figures/
    python -m scripts.paper.figures budget      # one figure by name

Figures: budget (RQ2), transfer (RQ4), judge_paired (RQ3), judge_curves, judge_retest, pilot_budget, universal_curves.
Each function prints the numbers it plots so they can be checked against the paper.
"""

import json
import statistics

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

import scripts.paper.figstyle as S

BUDGET = 2000
BENCHES = [("rjudge", "R-Judge"), ("atbench", "ATBench")]
JUDGE_BOUNDARY = 2.0


def load(path):
    return json.loads((S.RESULTS / path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in (S.RESULTS / path).read_text().split("\n") if line.strip()]


# ---------------------------------------------------------------- RQ2: flips by query budget
def _email_hits():
    hits = []
    base = S.RESULTS / "per_document" / "emails"
    for result in sorted(base.glob("campaign-*/*/result.json")):
        data = json.loads(result.read_text())
        if data.get("success"):
            hits.append(data["queries_at_hit"])
    return sorted(hits)


def _dataset_hits(key):
    out, n = [], 0
    for path in sorted((S.RESULTS / "per_document" / "datasets" / key / "asr").glob("yes_*.json")):
        record = json.loads(path.read_text())
        n += 1
        if record.get("success"):
            out.append(record["hit_at"])
    return sorted(out), n


COLORS = {  # Paul Tol's muted qualitative scheme
    "trec": "#332288", "ag_news": "#88CCEE", "dbpedia_14": "#44AA99", "yahoo_answers": "#117733",
    "patent": "#999933", "arxiv_abstracts": "#DDCC77", "20newsgroups": "#CC6677", "lexglue_ecthr": "#882255",
    "arxiv_fulltext": "#AA4499",
}


def budget():
    data = {key: _dataset_hits(key) for key, *_ in S.DATASETS}
    pooled = sorted(v for values, _ in data.values() for v in values)
    email_hits = _email_hits()

    def curve(values, n):
        xs = list(range(0, BUDGET + 1, 5))
        return xs, [100 * sum(v <= x for v in values) / n for x in xs]

    fig, ax = plt.subplots(figsize=(S.SINGLE, 2.25), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    for key, _name, short, _length in S.DATASETS:
        values, n = data[key]
        ax.plot(*curve(values, n), drawstyle="steps-post", color=COLORS[key], lw=1.3, label=short)
    ax.plot(*curve(pooled, 180), drawstyle="steps-post", color=S.INK, lw=2.0, label="all nine")
    ax.plot(*curve(email_hits, 28), drawstyle="steps-post", color=S.INK, lw=1.2, ls=(0, (3, 1.6)), label="email")
    ax.set_xlim(0, BUDGET)
    ax.set_ylim(0, 70)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("queries")
    ax.set_ylabel("targets flipped (%)")
    fig.legend(loc="outside lower center", ncols=4, fontsize=8, handlelength=1.5, columnspacing=1.0)
    S.save(fig, "fig_budget")
    print("  per-dataset confirmed flips: " + ", ".join(f"{s}={len(data[k][0])}/{data[k][1]}" for k, _, s, _ in S.DATASETS))
    print(f"  pooled {len(pooled)}/180; email {len(email_hits)}/28")


# ---------------------------------------------------------------- RQ4: transfer
def _luminance(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _text_color(rgb):
    lum = _luminance(rgb)
    ink = _luminance(to_rgb(S.INK))
    return "white" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / (ink + 0.05)) else S.INK


def transfer():
    matrix = load("transfer/datasets/matrix.json")
    keys = [k for k, *_ in S.DATASETS]
    short = [s for _, _, s, _ in S.DATASETS]
    n = len(keys)
    grid = np.full((n, n), np.nan)
    for i, src in enumerate(keys):
        for j, tgt in enumerate(keys):
            if i != j:
                grid[i, j] = 100 * matrix[src][tgt]
    row_mean, col_mean = np.nanmean(grid, axis=1), np.nanmean(grid, axis=0)
    cells = grid[~np.isnan(grid)]
    overall = cells.mean()
    cmap = plt.get_cmap("Oranges")
    gap = 0.32
    pos = list(range(n)) + [n - 1 + 1 + gap]
    fig, ax = plt.subplots(figsize=(S.SINGLE, 2.95), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)

    def cell(x, y, value, diagonal=False):
        if diagonal:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#E6E6E6", edgecolor="white", lw=1.0))
            ax.text(x, y, "–", ha="center", va="center", fontsize=S.ANNOT, color=S.MUTED)
            return
        rgb = to_rgb(cmap(value / 100))
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=rgb, edgecolor="white", lw=1.0))
        ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=S.ANNOT, color=_text_color(rgb),
                fontweight="bold" if round(value) >= 50 else "normal")

    for i in range(n):
        for j in range(n):
            cell(pos[j], pos[i], grid[i, j], diagonal=(i == j))
        cell(pos[n], pos[i], row_mean[i])
    for j in range(n):
        cell(pos[j], pos[n], col_mean[j])
    cell(pos[n], pos[n], overall)
    ax.set_xlim(-0.5, pos[n] + 0.5)
    ax.set_ylim(pos[n] + 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(pos, short + ["mean"], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(pos, short + ["mean"])
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("applied to the held-out documents of")
    ax.set_ylabel("suffix found on")
    S.save(fig, "fig_transfer")
    print(f"  mean {overall:.1f}%, cells >= 20: {(cells >= 20).sum()}, >= 50: {(cells >= 50).sum()}, n={cells.size}")


# ---------------------------------------------------------------- RQ3: trajectory held-out scores
def _per_trajectory(bench):
    records = {}
    for name in ("heldout_records.json", "heldout_records_controls_extra.json"):
        records.update(load(f"universal/trajectories/{bench}/{name}"))
    means = {}
    for condition in ("no_suffix", "suffix", "neutral_long"):
        scores = {}
        for r in records[condition]:
            if r.get("status") == "ok":
                scores.setdefault(r["case_id"], []).append(r["detail"]["score"])
        means[condition] = {case: statistics.fmean(v) for case, v in scores.items()}
    return means


def judge_paired():
    series = [("neutral_long", "neutral filler, same length", S.CONTROL), ("suffix", "universal suffix", S.ATTACK)]
    lo, hi = 1.2, 3.8
    fig, axes = plt.subplots(1, 2, figsize=(S.SINGLE, 1.95), layout="constrained", sharey=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    for ax, (bench, title) in zip(axes, [("atbench", "ATBench"), ("rjudge", "R-Judge")]):
        means = _per_trajectory(bench)
        clean = means["no_suffix"]
        ax.plot([lo, hi], [lo, hi], color=S.CONTROL, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(JUDGE_BOUNDARY, color=S.INK, lw=0.8, zorder=1)
        for key, label, color in series:
            cases = [c for c in clean if c in means[key]]
            ax.scatter([clean[c] for c in cases], [means[key][c] for c in cases], s=11, color=color, lw=0,
                       alpha=0.9, label=label, zorder=3 if key == "suffix" else 2)
            drop = statistics.fmean(clean[c] - means[key][c] for c in cases)
            print(f"  {title:8s} {label:28s} mean drop {drop:.3f}; below 2.0 {sum(means[key][c] < 2.0 for c in cases)}/{len(cases)}")
        ax.set_title(title, fontweight="normal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("without text")
    axes[0].set_ylabel("with text")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncols=2, fontsize=8, handletextpad=0.2,
               columnspacing=1.2, markerscale=1.3)
    S.save(fig, "fig_judge_paired")


# ---------------------------------------------------------------- trajectory training curves (appendix)
def _best_curve(bench):
    best, xs, ys = 0.0, [], []
    for r in read_jsonl(f"universal/trajectories/{bench}/training_curve.jsonl"):
        if r["phase"] == "search" and r["fitness"] is not None:
            best = max(best, r["fitness"])
        xs.append(r["query"])
        ys.append(best)
    return xs, ys


def judge_curves():
    styles = {"atbench": ("ATBench", S.ATTACK, "-", "o"), "rjudge": ("R-Judge", S.OTHER, (0, (4, 1.6)), "s")}
    fig, ax = plt.subplots(figsize=(S.SINGLE, 2.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    for key in ("atbench", "rjudge"):
        label, color, ls, marker = styles[key]
        xs, ys = _best_curve(key)
        uni = load(f"universal/trajectories/{key}/universal.json")
        marks = [i for i, x in enumerate(xs) if x % 500 == 0]
        ax.plot(xs, ys, color=color, ls=ls, lw=1.8, drawstyle="steps-post", marker=marker, ms=4.2, markevery=marks,
                mfc=color, mec="white", mew=0.5, label=label, zorder=3)
        if uni["search_success"]:
            i = next(i for i, y in enumerate(ys) if y > 0.5)
            ax.plot(xs[i], ys[i], marker="o", ms=5, mfc=S.ATTACK, mec=S.INK, mew=0.8, ls="none", zorder=4)
            ax.annotate(f"crosses at {xs[i]:,}", (xs[i], ys[i]), xytext=(-6, 5), textcoords="offset points",
                        ha="right", va="bottom", fontsize=S.ANNOT, color=S.INK)
        print(f"  {label}: {len(xs)} queries, best {max(ys):.3f}, success {uni['search_success']}")
    ax.axhline(0.5, color=S.CONTROL, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.text(30, 0.505, "threshold", ha="left", va="bottom", fontsize=S.ANNOT, color=S.MUTED)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0.25, 0.56)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("candidates")
    ax.set_ylabel("best-so-far mean P(safe),\n10 training trajectories")
    ax.legend(loc="lower right", handlelength=2.4)
    S.save(fig, "fig_judge_curves")


# ---------------------------------------------------------------- trajectory re-test (appendix)
def judge_retest():
    retest = load("per_document/trajectories/retest.json")
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.4), layout="constrained", sharey=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    for ax, (bench, title) in zip(axes, BENCHES):
        entries = sorted(retest["results"][bench], key=lambda e: e["texts"]["best_found"]["attacked_mean"])
        clean = [e["texts"]["best_found"]["clean_mean"] for e in entries]
        attacked = [e["texts"]["best_found"]["attacked_mean"] for e in entries]
        xs = range(len(entries))
        for x, c, a in zip(xs, clean, attacked):
            ax.plot([x, x], [a, c], color=S.CONTROL_LIGHT, lw=0.9, zorder=1)
        ax.plot(xs, clean, ls="none", marker="o", ms=5, mfc="white", mec=S.CONTROL, mew=1.1, zorder=2)
        ax.plot(xs, attacked, ls="none", marker="o", ms=5, mfc=S.ATTACK, mec=S.ATTACK, mew=0.6, zorder=3)
        ax.axhline(JUDGE_BOUNDARY, color=S.CONTROL, lw=0.9, ls=(0, (4, 2.5)), zorder=0)
        below = sum(a < JUDGE_BOUNDARY for a in attacked)
        ax.text(len(entries) - 0.6, JUDGE_BOUNDARY + 0.03, "verdict threshold", ha="right", va="bottom",
                fontsize=S.ANNOT, color=S.MUTED)
        ax.text(len(entries) - 0.6, JUDGE_BOUNDARY - 0.06, f"{below} of {len(entries)} targets below 2.0 with the suffix",
                ha="right", va="top", fontsize=S.ANNOT, color=S.INK)
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.8, len(entries) - 0.2)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("targets")
        print(f"  {title}: mean drop {statistics.fmean(c - a for c, a in zip(clean, attacked)):.2f}, below 2.0 {below}/{len(entries)}")
    axes[0].set_ylim(1.7, 3.4)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[0].set_ylabel("mean safety score\n(10 fresh queries)")
    handles = [Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=S.CONTROL, mew=1.1, label="clean trajectory"),
               Line2D([], [], ls="none", marker="o", ms=5, mfc=S.ATTACK, mec=S.ATTACK, label="with its best suffix")]
    fig.legend(handles=handles, loc="outside upper center", ncols=2)
    S.save(fig, "fig_judge_retest")


# ---------------------------------------------------------------- email searches by budget (appendix)
def _email_runs():
    out = []
    base = S.RESULTS / "per_document" / "emails"
    for result in sorted(base.glob("campaign-*/*/result.json")):
        config = json.loads((result.parent / "config.json").read_text())
        data = json.loads(result.read_text())
        out.append({"seed": config["seed_attack"], "hit": data.get("queries_at_hit") if data.get("success") else None})
    return out


def pilot_budget():
    data = _email_runs()
    later = ("question", "field", "computed", "confidence")
    groups = [("command", [r for r in data if r["seed"] == "command"], S.CONTROL, (0, (1.2, 1.6)), "D", 1.4),
              ("claim", [r for r in data if r["seed"] == "claim"], S.SEARCH, (0, (5, 2)), "s", 1.4),
              ("later seeds", [r for r in data if r["seed"] in later], S.OTHER, (0, (6, 1.5, 1.5, 1.5)), "^", 1.4),
              ("all", data, S.ATTACK, "-", "o", 2.4)]

    def curve(hits, n):
        xs, ys = [0], [0.0]
        for i, h in enumerate(sorted(hits), 1):
            xs += [h, h]
            ys += [ys[-1], 100 * i / n]
        return xs + [BUDGET], ys + [ys[-1]]

    fig, ax = plt.subplots(figsize=(S.SINGLE, 2.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    handles, labels = [], []
    for label, members, color, ls, marker, lw in groups:
        hits = [r["hit"] for r in members if r["hit"] is not None]
        xs, ys = curve(hits, len(members))
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=4 if label == "all" else 3)
        ax.plot(sorted(hits), [100 * i / len(members) for i in range(1, len(hits) + 1)], ls="none", marker=marker,
                ms=4.2 if label == "all" else 4, mfc=color, mec="white" if label == "all" else color, mew=0.6,
                zorder=5 if label == "all" else 4)
        handles.append(Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=4, mfc=color,
                              mec="white" if label == "all" else color, mew=0.6))
        labels.append(f"{label}: {len(hits)} of {len(members)}")
    order = [3, 1, 2, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", handlelength=2.6)
    ax.set_xlim(0, BUDGET)
    ax.set_ylim(0, 60)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("query budget")
    ax.set_ylabel("searches with a\nconfirmed flip (%)")
    S.save(fig, "fig_pilot_budget")
    print("  " + "; ".join(labels))


# ---------------------------------------------------------------- universal search curves (appendix)
def _series(entry):
    fitness = np.array([np.nan if v is None else v for v in entry["fitness"]], float)
    best = np.empty_like(fitness)
    changes = entry["best_changes"] + [[len(fitness) + 1, None]]
    for (q0, v), (q1, _) in zip(changes, changes[1:]):
        best[q0 - 1:q1 - 1] = np.nan if v is None else v
    best = np.fmax.accumulate(np.nan_to_num(best, nan=0.0))
    return fitness, best


def universal_curves():
    data = load("universal/datasets/training_curves.json")
    fig, axes = plt.subplots(3, 3, figsize=(S.FULL, 4.3), layout="constrained", sharex=True, sharey=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
    for ax, (key, name, _short, _tok) in zip(axes.ravel(), S.DATASETS):
        fitness, best = _series(data[key])
        x = np.arange(1, len(fitness) + 1)
        ax.scatter(x, fitness, s=1.2, color=S.CONTROL_LIGHT, linewidths=0, rasterized=True, zorder=1)
        ax.axhline(0.5, color=S.CONTROL, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
        ax.plot(x, best, color=S.ATTACK, lw=1.6, drawstyle="steps-post", zorder=3)
        if load(f"universal/datasets/{key}/universal.json")["search_success"]:
            rises = np.flatnonzero(np.diff(best) > 0) + 1
            q = int(rises[-1])
            ax.plot(q + 1, float(best[q]), marker="o", ms=5, mfc=S.ATTACK, mec=S.INK, mew=0.8, ls="none", zorder=4)
        if key == S.DATASETS[0][0]:
            ax.text(4950, 0.512, "threshold", ha="right", va="bottom", fontsize=S.ANNOT, color=S.MUTED)
        ax.set_title(name, loc="left")
    ax = axes[0, 0]
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 0.66)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    for a in axes.ravel():
        a.grid(axis="x", visible=False)
    fig.supxlabel("candidates")
    fig.supylabel("mean P(no), 10 training docs")
    handles = [Line2D([], [], marker="o", ls="none", ms=3, mfc=S.CONTROL_LIGHT, mec="none", label="one candidate suffix"),
               Line2D([], [], color=S.ATTACK, lw=1.6, label="best so far"),
               Line2D([], [], marker="o", ls="none", ms=5, mfc=S.ATTACK, mec=S.INK, mew=0.8, label="confirmed crossing")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3)
    S.save(fig, "fig_universal_curves")


FIGURES = {"budget": budget, "transfer": transfer, "judge_paired": judge_paired, "judge_curves": judge_curves,
           "judge_retest": judge_retest, "pilot_budget": pilot_budget, "universal_curves": universal_curves}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", choices=list(FIGURES) + [[]], default=[],
                        help="which figures to regenerate (default: all)")
    args = parser.parse_args()
    for name in args.names or list(FIGURES):
        print(name)
        FIGURES[name]()


if __name__ == "__main__":
    main()
