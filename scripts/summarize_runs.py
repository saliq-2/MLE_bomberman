"""Aggregate finished evaluation runs into one machine-readable summary.

Every comparison in this log so far has been assembled by reading numbers off
the eval script's console output and retyping them into a markdown table. That
is where transcription errors come from, and it is also why the figures could
not be generated automatically -- the aggregates existed only in prose.

This script reads the artifacts already on disk (results/<tag>.json plus
logs_by_tag/<tag>/game.log) for a set of run tags, reuses Experiment 24's
reprocessing path so the win/draw/loss logic is the corrected one, and writes
a single JSON holding per-seed values and per-config aggregates. No games are
re-run. scripts/make_report_figures.py reads that JSON, so a figure and the
table it illustrates come from the same numbers by construction.

Usage:
    python scripts/summarize_runs.py --out results/summary_exp26.json \
        --run "SARSA escape-robust:sarsa_agent:exp26_sarsa_escape_robust" \
        --run "SARSA baseline:sarsa_agent:exp26_sarsa_baseline32"

Each --run is "label:agent:tag_prefix". Configurations and seeds default to
the ones Experiments 23/25/26 use; missing runs are reported and skipped
rather than aborting the whole summary.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_opponents import REPO_ROOT, CONFIGS  # noqa: E402
from reprocess_exp24 import reprocess_one  # noqa: E402

METRICS = [
    "score_per_round", "win_rate", "self_kills", "deaths_by_opponent",
    "opponents_killed", "bombs_per_round", "crates_per_round",
    "coins_per_round", "avg_steps", "w", "d", "l",
]


def agg(per_seed, key):
    vals = [r[key] for r in per_seed]
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "per_seed": vals,
    }


def summarize_run(label, agent, prefix, seeds, configs):
    out = {"label": label, "agent": agent, "tag_prefix": prefix, "configs": {}}
    for config_name, opponents in configs.items():
        per_seed, missing = [], []
        for seed in seeds:
            tag = prefix + "_" + config_name + "_seed" + str(seed)
            stats_path = os.path.join(REPO_ROOT, "results", tag + ".json")
            log_path = os.path.join(REPO_ROOT, "logs_by_tag", tag, "game.log")
            if not (os.path.exists(stats_path) and os.path.exists(log_path)):
                missing.append(tag)
                continue
            try:
                per_seed.append(reprocess_one(agent, opponents, tag))
            except Exception as exc:
                missing.append(tag + " (" + type(exc).__name__ + ")")
        if not per_seed:
            print("  -- " + label + " [" + config_name + "]: no completed seeds, skipped")
            continue
        entry = {m: agg(per_seed, m) for m in METRICS}
        entry["n_seeds"] = len(per_seed)
        entry["missing"] = missing
        # The cross-check that every table in this log reports: the score
        # reconstructed from game.log must match --save-stats' cumulative
        # total. A False here invalidates the win/draw/loss numbers for that
        # seed, so it is carried into the summary rather than printed once.
        entry["score_check_ok"] = all(r["score_check_ok"] for r in per_seed)
        out["configs"][config_name] = entry
        print("  " + label + " [" + config_name + "]: n=" + str(len(per_seed))
              + "  score/round=" + format(entry["score_per_round"]["mean"], ".3f")
              + "±" + format(entry["score_per_round"]["sd"], ".3f")
              + "  win=" + format(100 * entry["win_rate"]["mean"], ".1f") + "%"
              + "  self_kills=" + format(entry["self_kills"]["mean"], ".1f")
              + "  check_ok=" + str(entry["score_check_ok"])
              + ("  MISSING=" + str(len(missing)) if missing else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='"label:agent:tag_prefix", repeatable')
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=300)
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    args = ap.parse_args()

    seeds = [args.seed_offset + i for i in range(args.seeds)]
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = {k: v for k, v in CONFIGS.items() if k in wanted}
    if not configs:
        print("no valid configs selected")
        return 1

    runs = []
    for spec in args.run:
        parts = spec.split(":")
        if len(parts) != 3:
            print('bad --run (want "label:agent:tag_prefix"): ' + spec)
            return 1
        label, agent, prefix = (p.strip() for p in parts)
        runs.append(summarize_run(label, agent, prefix, seeds, configs))

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"seeds": seeds, "runs": runs}, fh, indent=1)
    print("wrote " + os.path.relpath(out_path, REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
