"""Reprocess Experiment 24's existing results/*.json + logs_by_tag/*/game.log
with the fixed win_draw_loss (see disambiguate_names in eval_vs_opponents.py)
-- no games are rerun, only the win/draw/loss classification is recomputed
against data already on disk from the corrected-log-dir run.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_opponents import (REPO_ROOT, CONFIGS, parse_game_log,
                                win_draw_loss, disambiguate_names)


def reprocess_one(agent, opponents, tag):
    log_path = os.path.join(REPO_ROOT, "logs_by_tag", tag, "game.log")
    stats_path = os.path.join(REPO_ROOT, "results", f"{tag}.json")
    with open(stats_path) as f:
        data = json.load(f)

    all_names = disambiguate_names([agent] + opponents)
    rounds_log, self_kills, deaths_by_opp, opp_killed = parse_game_log(log_path, agent, all_names)
    w, d, l = win_draw_loss(rounds_log, agent, all_names)

    lifetime = data["by_agent"][agent]
    n = lifetime["rounds"]
    round_stats = list(data["by_round"].values())
    score_from_log = sum(r.get(agent, 0) for r in rounds_log)
    score_check_ok = (score_from_log == lifetime["score"])

    return {
        "score_per_round": lifetime["score"] / n,
        "w": w, "d": d, "l": l,
        "win_rate": w / n,
        "self_kills": self_kills,
        "deaths_by_opponent": deaths_by_opp,
        "opponents_killed": opp_killed,
        "bombs_per_round": lifetime.get("bombs", 0) / n,
        "crates_per_round": lifetime.get("crates", 0) / n,
        "coins_per_round": lifetime.get("coins", 0) / n,
        "avg_steps": sum(r["steps"] for r in round_stats) / n,
        "score_check_ok": score_check_ok,
        "n_rounds": n,
    }


def main():
    agent_checkpoints = {
        "qlearn_agent": ["exp24_qlearn_vs_peaceful_coin", "exp24_qlearn_vs_rule_based"],
        "sarsa_agent": ["exp24_sarsa_vs_peaceful_coin", "exp24_sarsa_vs_rule_based"],
    }
    seeds = range(5)

    for agent, tags in agent_checkpoints.items():
        for base_tag in tags:
            for config_name, opponents in CONFIGS.items():
                per_seed = []
                for seed in seeds:
                    tag = f"{base_tag}_{config_name}_seed{seed}"
                    per_seed.append(reprocess_one(agent, opponents, tag))

                def agg(key):
                    vals = [r[key] for r in per_seed]
                    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

                w_m, w_sd = agg("w")
                d_m, d_sd = agg("d")
                l_m, l_sd = agg("l")
                wr_m, wr_sd = agg("win_rate")
                print(f"{base_tag} [{config_name}]: "
                      f"W={w_m:.1f}±{w_sd:.1f} D={d_m:.1f}±{d_sd:.1f} L={l_m:.1f}±{l_sd:.1f} "
                      f"win_rate={wr_m:.3f}±{wr_sd:.3f} "
                      f"(score_check_ok all: {all(r['score_check_ok'] for r in per_seed)})")
                for r in per_seed:
                    pass
        print()


if __name__ == "__main__":
    main()
