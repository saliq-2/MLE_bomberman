"""Experiment 24: train against opponents, then evaluate under Experiment
23's exact protocol (same three configs, 200 rounds, 5 seeds), reusing its
methodology -- --save-stats only gives cumulative per-agent totals, not a
per-round/per-agent breakdown (confirmed: environment.py's round_statistics
sums across *all* agents), so win/draw/loss and the self-kill vs
deaths-by-opponent split are reconstructed from game.log, cross-checked
against the JSON's cumulative score.

This is an independent re-implementation matching Experiment 23's described
approach (log line formats verified directly against environment.py), not
a copy of an undisclosed prior script.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get("BOMBERMAN_PYTHON", sys.executable)

# main.py's default --log-dir is a single shared <repo>/logs, opened with
# mode="w" per subprocess. Two eval_vs_opponents.py invocations running in
# parallel (e.g. one per agent, as we do to save wall-clock time) both write
# game.log there concurrently and silently corrupt each other's log --
# discovered via score_check_ok being False on *every* seed/config in a run
# where qlearn's and sarsa's jobs overlapped the whole time. Each invocation
# now gets its own log dir, keyed by --tag, so parallel jobs can't collide.

COIN_RE = re.compile(r"Agent <([^>]+)> picked up coin")
SELF_KILL_RE = re.compile(r"Agent <([^>]+)> blown up by own bomb")
OPP_KILL_RE = re.compile(r"Agent <([^>]+)> blown up by agent <([^>]+)>'s bomb")
ROUND_START_RE = re.compile(r"STARTING ROUND #(\d+)")
ROUND_END_RE = re.compile(r"WRAPPING UP ROUND #(\d+)")
REWARD_COIN = 1
REWARD_KILL = 5


def _run(args, env=None):
    subprocess.run(args, check=True, cwd=REPO_ROOT, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def train(agent, opponents, rounds, seed, checkpoint_name, scenario="classic"):
    model_file = os.path.join(REPO_ROOT, "agent_code", agent, "model.pt")
    if os.path.exists(model_file):
        os.remove(model_file)
    env = os.environ.copy()
    env["QLEARN_SEED"] = str(seed)
    _run([PYTHON, "main.py", "play", "--no-gui",
          "--agents", agent, *opponents, "--train", "1",
          "--scenario", scenario, "--n-rounds", str(rounds),
          "--seed", str(seed), "--continue-without-training"], env)
    saved_as = os.path.join(REPO_ROOT, "agent_code", agent, checkpoint_name)
    os.replace(model_file, saved_as)
    return saved_as


def parse_game_log(log_path, our_name, all_names):
    """Returns (rounds: list[dict[name, score_this_round]], self_kills,
    deaths_by_opponent, opponents_killed_by_us)."""
    rounds = []
    cur = defaultdict(int)
    self_kills = deaths_by_opponent = opponents_killed = 0
    with open(log_path) as f:
        for line in f:
            if ROUND_START_RE.search(line):
                cur = defaultdict(int)
                continue
            if ROUND_END_RE.search(line):
                rounds.append(dict(cur))
                continue
            m = COIN_RE.search(line)
            if m:
                cur[m.group(1)] += REWARD_COIN
                continue
            m = SELF_KILL_RE.search(line)
            if m:
                if m.group(1) == our_name:
                    self_kills += 1
                continue
            m = OPP_KILL_RE.search(line)
            if m:
                victim, killer = m.group(1), m.group(2)
                cur[killer] += REWARD_KILL
                if victim == our_name:
                    deaths_by_opponent += 1
                if killer == our_name:
                    opponents_killed += 1
                continue
    return rounds, self_kills, deaths_by_opponent, opponents_killed


def win_draw_loss(rounds, our_name, all_names):
    w = d = l = 0
    for r in rounds:
        scores = {name: r.get(name, 0) for name in all_names}
        our_score = scores[our_name]
        max_score = max(scores.values())
        leaders = [n for n, sc in scores.items() if sc == max_score]
        if our_score < max_score:
            l += 1
        elif len(leaders) == 1:
            w += 1
        else:
            d += 1
    return w, d, l


def disambiguate_names(agent_dirs):
    """Reproduce environment.py's setup_agents naming exactly: when an
    agent_dir appears more than once in the CLI --agents list (e.g. three
    rule_based_agent instances in the tournament config), each occurrence is
    suffixed _0, _1, _2 (first-seen order) in game.log and --save-stats.
    Passing the raw (possibly duplicated) CLI list as all_names into
    win_draw_loss silently collapses every duplicate into one dict key that
    never matches a real per-round key -- caught only because L was 0 on
    every single tournament-config round-set, not because of a crash."""
    total = {}
    for d in agent_dirs:
        total[d] = total.get(d, 0) + 1
    seen = {}
    result = []
    for d in agent_dirs:
        if total[d] > 1:
            i = seen.get(d, 0)
            result.append(f"{d}_{i}")
            seen[d] = i + 1
        else:
            result.append(d)
    return result


def evaluate_one(agent, checkpoint_path, opponents, n_rounds, seed, tag, scenario="classic"):
    model_file = os.path.join(REPO_ROOT, "agent_code", agent, "model.pt")
    import shutil
    shutil.copy(checkpoint_path, model_file)

    log_dir = os.path.join(REPO_ROOT, "logs_by_tag", tag)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "game.log")

    env = os.environ.copy()
    env["QLEARN_SEED"] = str(seed)
    stats_path = os.path.join("results", f"{tag}.json")
    _run([PYTHON, "main.py", "play", "--no-gui",
          "--agents", agent, *opponents,
          "--scenario", scenario, "--n-rounds", str(n_rounds),
          "--seed", str(seed), "--continue-without-training",
          "--save-stats", stats_path, "--match-name", tag,
          "--log-dir", log_dir], env)

    with open(os.path.join(REPO_ROOT, stats_path)) as f:
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
        "suicide_rate": sum(r.get("suicides", 0) for r in round_stats) / n,
        "avg_steps": sum(r["steps"] for r in round_stats) / n,
        "score_check_ok": score_check_ok,
        "score_from_log": score_from_log,
        "score_from_json": lifetime["score"],
        "n_rounds": n,
    }


CONFIGS = {
    "task3": ["peaceful_agent", "coin_collector_agent"],
    "task4": ["rule_based_agent"],
    "tournament": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--checkpoint", required=True, help="path to the .pt checkpoint to evaluate")
    p.add_argument("--n-rounds", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    for config_name, opponents in CONFIGS.items():
        per_seed = []
        for i in range(args.seeds):
            seed = args.seed_offset + i
            tag = f"{args.tag}_{config_name}_seed{seed}"
            result = evaluate_one(args.agent, args.checkpoint, opponents, args.n_rounds, seed, tag)
            per_seed.append(result)
            print(f"[{config_name}] seed={seed} score/round={result['score_per_round']:.3f} "
                  f"W/D/L={result['w']}/{result['d']}/{result['l']} "
                  f"self_kills={result['self_kills']} deaths_by_opp={result['deaths_by_opponent']} "
                  f"opp_killed={result['opponents_killed']} "
                  f"bombs/round={result['bombs_per_round']:.2f} "
                  f"crates/round={result['crates_per_round']:.2f} "
                  f"coins/round={result['coins_per_round']:.3f} "
                  f"suicide_rate={result['suicide_rate']:.3f} "
                  f"score_check_ok={result['score_check_ok']}")

        def agg(key):
            vals = [r[key] for r in per_seed]
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        print(f"=== {config_name} aggregate over {args.seeds} seeds ===")
        for key in ["score_per_round", "win_rate", "self_kills", "deaths_by_opponent",
                    "opponents_killed", "bombs_per_round", "crates_per_round",
                    "coins_per_round", "suicide_rate", "avg_steps"]:
            m, sd = agg(key)
            print(f"  {key}: {m:.3f} +/- {sd:.3f}")
        all_ok = all(r["score_check_ok"] for r in per_seed)
        print(f"  score_check_ok (all seeds): {all_ok}")
        print()


if __name__ == "__main__":
    main()
