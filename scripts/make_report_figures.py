"""Regenerate every figure used in the final report, from results/*.json.

The point of having this as a script rather than a folder of hand-exported
images: each figure is reproducible from the run artifacts that are already
committed, so a number in the report and the picture of that number cannot
drift apart. Re-run it after any new experiment and the figures update.

Colours come from a categorical palette checked for colour-vision deficiency
separation rather than picked by eye (worst adjacent-pair CVD dE 9.1,
normal-vision dE 19.6, OKLab x100). Three of the slots sit under 3:1 contrast
on a white page, so every series is also labelled directly on the plot --
identity is never carried by colour alone, which matters for a report that may
well be printed in greyscale.

Deliberately no dual-axis plots: where two quantities of different scale belong
to the same comparison (coins and self-kills, score and win rate) they get
adjacent panels sharing an x-axis instead of two y-scales on one frame.

Run: python scripts/make_report_figures.py [--outdir figures]
"""
import argparse
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def load(pattern):
    """Load every results JSON matching a glob, sorted by name."""
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS, pattern)))]


def agent_of(doc):
    return next(iter(doc["by_agent"]))


def per_round(doc, key):
    a = doc["by_agent"][agent_of(doc)]
    return a.get(key, 0) / a["rounds"]


def mean_sd(vals):
    if not vals:
        return 0.0, 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def bars(ax, labels, means, sds, colors, fmt="{:.2f}", ylabel=""):
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.6, color=colors[:len(labels)],
           yerr=sds if any(sds) else None, capsize=3,
           error_kw={"ecolor": INK_SOFT, "elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    span = max(means) if max(means) else 1.0
    for xi, (m, sd) in enumerate(zip(means, sds)):
        ax.text(xi, m + sd + span * 0.04, fmt.format(m),
                ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
    top = max((m + s) for m, s in zip(means, sds)) or 1.0
    ax.set_ylim(0, top * 1.22)
    style(ax)


def fig_shaping_ablation(outdir):
    """Task 1: does reward shaping actually buy anything?"""
    groups = [
        ("With shaping", "task1_shaping_fixed_v2_seed*_eval.json"),
        ("Sparse reward only", "task1_noshaping_fixed_v2_seed*_eval.json"),
    ]
    labels, means, sds = [], [], []
    for label, pat in groups:
        docs = load(pat)
        if not docs:
            return None
        m, sd = mean_sd([per_round(d, "coins") for d in docs])
        labels.append(label + "\n(n=" + str(len(docs)) + " seeds)")
        means.append(m)
        sds.append(sd)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    bars(ax, labels, means, sds, SERIES, ylabel="Coins collected per round")
    ax.set_title("Task 1: reward shaping vs. sparse reward\ncoin-heaven, greedy evaluation")
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_task1_shaping_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_training_curves(outdir, window=50):
    """Task 2 training progress across the changes that moved it.

    Two panels rather than one, because the self-kill curve on its own is
    actively misleading: every stage sits at 0.7-0.95 throughout, which reads
    as "nothing improved" when the truth is that a policy which never drops a
    bomb cannot kill itself either. Coins-per-round is what separates the
    stages; the self-kill panel is the cost side of the same trade, and the
    two are only interpretable together.
    """
    stages = [
        ("Exp 13", "on-target fix", "exp13_train_seed0.json"),
        ("Exp 15", "escape shaping", "exp15_train_seed0.json"),
        ("Exp 16", "reward fixes", "exp16_train_seed0.json"),
        ("Exp 17", "gamma sweep", "exp17_train_seed0.json"),
        ("Exp 18", "escape commitment", "exp18_train_seed0.json"),
    ]
    loaded = []
    for i, (short, desc, fname) in enumerate(stages):
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            continue
        doc = json.load(open(path))
        rounds = list(doc["by_round"].values())
        if len(rounds) < window:
            continue
        loaded.append((short, desc, SERIES[i % len(SERIES)], rounds))
    if not loaded:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    kernel = np.ones(window) / window
    panels = [
        (axes[0], "coins", "Coins per round  (higher is better)", None),
        (axes[1], "suicides", "Self-kill rate  (lower is better)", (0, 1.05)),
    ]
    for ax, key, title, ylim in panels:
        ends = []
        for short, desc, colour, rounds in loaded:
            vals = [r.get(key, 0) for r in rounds]
            roll = np.convolve(vals, kernel, mode="valid")
            xs = np.arange(len(roll)) + window
            ax.plot(xs, roll, color=colour, linewidth=1.8,
                    label=short + "  " + desc)
            ends.append([roll[-1], short, colour, xs[-1]])

        # Nudge the end-of-line labels apart. Several of these curves finish
        # within a hair of each other, and overlapping labels are worse than
        # no labels -- but the labels are also the relief for three palette
        # slots that sit under 3:1 contrast on white, so dropping them is not
        # an option either. Spread them over a minimum gap instead.
        span = (ylim[1] - ylim[0]) if ylim else max(e[0] for e in ends) or 1.0
        gap = span * 0.055
        ends.sort(key=lambda e: e[0])
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < gap:
                ends[j][0] = ends[j - 1][0] + gap
        for y, short, colour, x_end in ends:
            ax.text(x_end + 10, y, short, color=colour, fontsize=7.5,
                    va="center", fontweight="bold")

        ax.set_xlabel("Training round")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.14)
        if ylim:
            ax.set_ylim(*ylim)
        style(ax)
    axes[0].set_ylabel("Coins per round (rolling mean, " + str(window) + ")")
    axes[1].set_ylabel("Self-kills per round (rolling mean, " + str(window) + ")")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Task 2 training progress: a policy that never bombs cannot kill itself either",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_task2_training_curves.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_task2_progression(outdir):
    """Task 2 greedy-eval metrics across the experiment sequence."""
    stages = [
        ("Exp 10", "task2_exp10_seed*_eval.json"),
        ("Exp 11", "task2_exp11_seed*_eval.json"),
        ("Exp 12", "task2_exp12_seed*_eval.json"),
        ("Exp 13", "task2_exp13_seed*_eval.json"),
        ("Exp 18", "task2_exp18_seed*_eval.json"),
        ("Exp 20", "task2_exp20_10seed_seed*_eval.json"),
        ("Exp 22", "task2_exp22_seed*_eval.json"),
    ]
    labels, coin_m, coin_s, sui_m, sui_s = [], [], [], [], []
    for label, pat in stages:
        docs = load(pat)
        if not docs:
            continue
        labels.append(label)
        m, sd = mean_sd([per_round(d, "coins") for d in docs])
        coin_m.append(m)
        coin_s.append(sd)
        m, sd = mean_sd([per_round(d, "suicides") for d in docs])
        sui_m.append(m)
        sui_s.append(sd)
    if not labels:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    x = np.arange(len(labels))
    panels = [
        (coin_m, coin_s, "Coins per round  (higher is better)", "Coins per round", SERIES[0]),
        (sui_m, sui_s, "Self-kill rate  (lower is better)", "Self-kills per round", SERIES[1]),
    ]
    for ax, (m, sd, title, ylab, colour) in zip(axes, panels):
        ax.errorbar(x, m, yerr=sd, color=colour, linewidth=2,
                    marker="o", markersize=5, capsize=3,
                    ecolor=INK_SOFT, elinewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("Task 2: greedy-evaluation metrics across the experiment sequence",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_task2_progression.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_qlearn_vs_sarsa(outdir):
    """The two-model comparison, on both tasks."""
    panels = [
        ("Task 1  (coin-heaven)", "Coins per round", "coins",
         "task1_qlearn_10seed_seed*_eval.json", "task1_sarsa_seed*_eval.json"),
        ("Task 2  (classic)", "Coins per round", "coins",
         "task2_exp22_seed*_eval.json", "task2_sarsa_seed*_eval.json"),
        ("Task 2  (classic)", "Self-kills per round", "suicides",
         "task2_exp22_seed*_eval.json", "task2_sarsa_seed*_eval.json"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4))
    any_plotted = False
    for ax, (title, ylab, key, qpat, spat) in zip(axes, panels):
        qd, sdocs = load(qpat), load(spat)
        if not qd or not sdocs:
            ax.axis("off")
            continue
        any_plotted = True
        qm, qs = mean_sd([per_round(d, key) for d in qd])
        sm, ss = mean_sd([per_round(d, key) for d in sdocs])
        bars(ax, ["Q-learning\n(n=" + str(len(qd)) + ")", "SARSA\n(n=" + str(len(sdocs)) + ")"],
             [qm, sm], [qs, ss], SERIES, ylabel=ylab)
        ax.set_title(title, fontsize=9)
    if not any_plotted:
        plt.close(fig)
        return None
    fig.suptitle("Off-policy vs. on-policy under an identical feature set and reward table",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_qlearn_vs_sarsa.png")
    fig.savefig(p)
    plt.close(fig)
    return p


CONFIG_LABELS = {
    "task3": "Task 3\npeaceful + coin_collector",
    "task4": "Task 4\n1x rule_based",
    "tournament": "Tournament\n3x rule_based",
}


def fig_tournament_decision(outdir, summary="results/summary_exp27_decision.json"):
    """The submission decision: candidate checkpoints against the provided agents.

    Score per round and win rate are the two numbers the decision turns on and
    they live on different scales, so they get two panels rather than two
    y-axes on one frame.
    """
    path = os.path.join(REPO_ROOT, summary)
    if not os.path.exists(path):
        return None
    doc = json.load(open(path))
    runs = [r for r in doc["runs"] if r["configs"]]
    if not runs:
        return None

    configs = [c for c in CONFIG_LABELS if any(c in r["configs"] for r in runs)]
    if not configs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(max(7.6, 2.6 * len(configs) + 4.0), 4.5))
    x = np.arange(len(configs))
    width = min(0.8 / len(runs), 0.34)
    # With one configuration and several candidates the bars are narrow, so the
    # value labels are set upright only when there is room for them.
    rotate = 90 if len(runs) > 3 and len(configs) > 1 else 0
    panels = [
        (axes[0], "score_per_round", "Score per round", "{:.2f}", 1.0),
        (axes[1], "win_rate", "Win rate (%)", "{:.1f}", 100.0),
    ]
    for ax, key, ylab, fmt, scale in panels:
        top = 0.0
        for j, run in enumerate(runs):
            means, sds = [], []
            for c in configs:
                entry = run["configs"].get(c)
                means.append(entry[key]["mean"] * scale if entry else 0.0)
                sds.append(entry[key]["sd"] * scale if entry else 0.0)
            offs = (j - (len(runs) - 1) / 2) * width
            ax.bar(x + offs, means, width=width * 0.9, color=SERIES[j % len(SERIES)],
                   label=run["label"], yerr=sds if any(sds) else None, capsize=2.5,
                   error_kw={"ecolor": INK_SOFT, "elinewidth": 0.9})
            for xi, (m, sd) in enumerate(zip(means, sds)):
                ax.text(x[xi] + offs, m + sd + 0.03 * max(means or [1]), fmt.format(m),
                        ha="center", va="bottom", fontsize=7, color=INK,
                        fontweight="bold", rotation=rotate)
                top = max(top, m + sd)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, (top or 1.0) * 1.28)
        style(ax)

    # Legend under both panels rather than inside one of them: with five
    # candidates it is wide, and in-axes it sat on top of the tallest bar's
    # value label -- which is also the relief for the low-contrast palette
    # slots, so it is not something that can be allowed to be covered.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center",
               ncol=min(3, len(runs)), frameon=False, bbox_to_anchor=(0.5, 0.0))

    n = max((e["n_seeds"] for r in runs for e in r["configs"].values()), default=0)
    fig.suptitle("Submission candidates, tournament configuration\n"
                 "classic, 200 rounds x " + str(n) + " seeds, greedy evaluation",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.99))
    p = os.path.join(outdir, "fig5_tournament_decision.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_task34_curriculum(outdir):
    """Tasks 3 and 4 before and after the Experiment 28 curriculum.

    Three panels, not one: the point of this experiment is that the curriculum
    moves different quantities in different directions, and a single "is it
    better" bar would hide exactly that. Task 3's metric is kills (the brief
    says "hunt and blow up"), Task 4's is survival, and score is shown for both
    because it is what the curriculum costs.
    """
    t3p = os.path.join(REPO_ROOT, "results/summary_exp28_t3.json")
    t4p = os.path.join(REPO_ROOT, "results/summary_exp28_rb3.json")
    if not (os.path.exists(t3p) and os.path.exists(t4p)):
        return None
    t3 = json.load(open(t3p))["runs"]
    t4 = json.load(open(t4p))["runs"]

    def grab(runs, cfg, key):
        out = []
        for r in runs:
            e = r["configs"].get(cfg)
            out.append((e[key]["mean"], e[key]["sd"]) if e else (0.0, 0.0))
        return out

    panels = [
        ("Task 3: opponents killed\n(higher = hunting)",
         grab(t3, "task3", "opponents_killed"), "Kills per 200 rounds"),
        ("Task 4: total deaths\n(lower = holding your own)",
         None, "Deaths per 200 rounds"),
        ("Score per round\n(what the curriculum costs)",
         None, "Score per round"),
    ]
    sk = grab(t4, "task4", "self_kills")
    do = grab(t4, "task4", "deaths_by_opponent")
    panels[1] = (panels[1][0], [(sk[i][0] + do[i][0], (sk[i][1] ** 2 + do[i][1] ** 2) ** 0.5)
                                for i in range(len(sk))], panels[1][2])
    s3 = grab(t3, "task3", "score_per_round")
    s4 = grab(t4, "task4", "score_per_round")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6))
    for ax, (title, vals, ylab) in zip(axes[:2], panels[:2]):
        bars(ax, ["before", "after"], [v[0] for v in vals], [v[1] for v in vals],
             SERIES, fmt="{:.1f}", ylabel=ylab)
        ax.set_title(title, fontsize=8.5)

    ax = axes[2]
    x = np.arange(2)
    for j, (vals, lab) in enumerate([(s3, "Task 3"), (s4, "Task 4")]):
        off = (j - 0.5) * 0.36
        ax.bar(x + off, [v[0] for v in vals], width=0.32, color=SERIES[j],
               yerr=[v[1] for v in vals], capsize=3, label=lab,
               error_kw={"ecolor": INK_SOFT, "elinewidth": 1})
        for xi, v in enumerate(vals):
            ax.text(x[xi] + off, v[0] + v[1] + 0.05, "{:.2f}".format(v[0]),
                    ha="center", va="bottom", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["before", "after"])
    ax.set_ylabel(panels[2][2])
    ax.set_ylim(0, 2.0)
    ax.set_title(panels[2][0], fontsize=8.5)
    ax.legend(fontsize=7.5)
    style(ax)

    fig.suptitle("Experiment 28: one curriculum mechanism, opposite policies against weak and strong opponents",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    p = os.path.join(outdir, "fig6_task34_curriculum.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    made = []
    for fn in (fig_shaping_ablation, fig_training_curves,
               fig_task2_progression, fig_qlearn_vs_sarsa,
               fig_tournament_decision, fig_task34_curriculum):
        try:
            p = fn(args.outdir)
        except Exception as exc:
            print("  !! " + fn.__name__ + ": " + str(exc))
            continue
        if p:
            made.append(p)
            print("  wrote " + os.path.relpath(p, REPO_ROOT))
        else:
            print("  -- " + fn.__name__ + ": source results missing, skipped")
    print(str(len(made)) + " figure(s) written to " + os.path.relpath(args.outdir, REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
