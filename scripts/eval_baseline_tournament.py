"""Experiment 25 (part of): regenerate Experiment 23's tournament baseline
with the corrected disambiguate_names() win/draw/loss logic, using a
zero-padded copy of model_task2_baseline.pt (see docs/experiments.md --
the padding is provably behavior-equivalent to the true 23-feature policy:
Experiment 24 only appended feature indices 23-29, weights on those are 0).
Tournament config only, 200 rounds, 5 seeds -- matches Experiment 23/24's
protocol exactly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_opponents import evaluate_one, CONFIGS
import statistics

AGENTS = ["qlearn_agent", "sarsa_agent"]
SEEDS = range(5)
CONFIG_NAME = "tournament"
OPPONENTS = CONFIGS[CONFIG_NAME]

for agent in AGENTS:
    checkpoint = os.path.join("agent_code", agent, "model_task2_baseline_padded.pt")
    per_seed = []
    for seed in SEEDS:
        tag = f"exp25_{agent.split('_')[0]}_baseline_{CONFIG_NAME}_seed{seed}"
        result = evaluate_one(agent, checkpoint, OPPONENTS, 200, seed, tag, scenario="classic")
        per_seed.append(result)
        print(f"[{agent} baseline] seed={seed} score/round={result['score_per_round']:.3f} "
              f"W/D/L={result['w']}/{result['d']}/{result['l']} "
              f"self_kills={result['self_kills']} deaths_by_opp={result['deaths_by_opponent']} "
              f"opp_killed={result['opponents_killed']} "
              f"bombs/round={result['bombs_per_round']:.2f} "
              f"crates/round={result['crates_per_round']:.2f} "
              f"coins/round={result['coins_per_round']:.3f} "
              f"score_check_ok={result['score_check_ok']}")

    def agg(key):
        vals = [r[key] for r in per_seed]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"=== {agent} baseline tournament aggregate over {len(SEEDS)} seeds ===")
    for key in ["score_per_round", "win_rate", "self_kills", "deaths_by_opponent",
                "opponents_killed", "bombs_per_round", "crates_per_round",
                "coins_per_round", "avg_steps"]:
        m, sd = agg(key)
        print(f"  {key}: {m:.3f} +/- {sd:.3f}")
    all_ok = all(r["score_check_ok"] for r in per_seed)
    print(f"  score_check_ok (all seeds): {all_ok}")
    print()
