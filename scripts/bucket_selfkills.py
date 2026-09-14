"""Experiment 25, part 1 driver: aggregate classify_selfkills.py's bucket
split over 5 seeds for a given (agent, tag-prefix) tournament run."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_selfkills import classify_log

OPPONENTS = ["rule_based_agent_0", "rule_based_agent_1", "rule_based_agent_2"]


def run(agent, tag_prefix, label):
    total_a = total_b = total_unmatched = 0
    per_seed = []
    for seed in range(5):
        log_path = os.path.join("logs_by_tag", f"{tag_prefix}_seed{seed}", "game.log")
        results = classify_log(log_path, agent, OPPONENTS)
        a = sum(1 for r in results if r["bucket"] == "a")
        b = sum(1 for r in results if r["bucket"] == "b")
        u = sum(1 for r in results if r["bucket"] == "unmatched")
        per_seed.append((a, b, u))
        total_a += a
        total_b += b
        total_unmatched += u
    n = total_a + total_b + total_unmatched
    print(f"{label}: per-seed (a,b,unmatched) = {per_seed}")
    print(f"{label}: TOTAL a={total_a} b={total_b} unmatched={total_unmatched} "
          f"(n={n}, a%={100*total_a/n:.1f}%, b%={100*total_b/n:.1f}%)" if n else f"{label}: n=0")
    print()


if __name__ == "__main__":
    run("qlearn_agent", "exp24_qlearn_vs_rule_based_tournament", "qlearn vs_rule_based")
    run("sarsa_agent", "exp24_sarsa_vs_rule_based_tournament", "sarsa vs_rule_based")
    if os.path.isdir("logs_by_tag/exp25_qlearn_baseline_tournament_seed0"):
        run("qlearn_agent", "exp25_qlearn_baseline_tournament", "qlearn baseline")
    if os.path.isdir("logs_by_tag/exp25_sarsa_baseline_tournament_seed0"):
        run("sarsa_agent", "exp25_sarsa_baseline_tournament", "sarsa baseline")
