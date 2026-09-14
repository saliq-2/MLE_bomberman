"""Train/eval harness for repo experiments (see docs/experiments.md).

Trains a fresh model for N rounds, freezes it (runs without --train, so
callbacks.py uses epsilon=0 and the saved weights), evaluates over M rounds,
and reports completion rate / steps-conditional-on-completion separately
instead of one blended "avg steps" number. Repeats over several seeds and
prints mean +/- std, since a single run's per-block numbers are noise (see
docs/experiments.md Experiment 1's block-to-block spread).

Both the board (--seed) and the agent's own RNG (QLEARN_SEED env var,
consumed by agent_code/qlearn_agent/callbacks.py) are set explicitly per
repeat, so every run is fully reproducible. Train and eval use *different*
board seeds (eval = 10000 + repeat index) so eval rounds are never a replay
of boards seen during training. Both seeds are recorded into the saved
results JSON under "_meta".

Usage:
    python scripts/run_experiment.py --agent qlearn_agent --scenario coin-heaven \
        --max-target 50 --train-rounds 300 --eval-rounds 100 --seeds 5 \
        --tag task1_shaping
    QLEARN_USE_SHAPING=0 python scripts/run_experiment.py ... --tag task1_noshaping
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get("BOMBERMAN_PYTHON", sys.executable)
EVAL_SEED_OFFSET = 10_000


def _run(args, env):
    subprocess.run(args, check=True, cwd=REPO_ROOT, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def train_then_eval(agent, scenario, train_rounds, eval_rounds, tag, seed, env):
    model_file = os.path.join(REPO_ROOT, "agent_code", agent, "model.pt")
    if os.path.exists(model_file):
        os.remove(model_file)

    run_env = env.copy()
    run_env["QLEARN_SEED"] = str(seed)
    train_seed = seed
    eval_seed = EVAL_SEED_OFFSET + seed

    _run([PYTHON, "main.py", "play", "--no-gui", "--agents", agent, "--train", "1",
          "--scenario", scenario, "--n-rounds", str(train_rounds),
          "--seed", str(train_seed),
          "--continue-without-training"], run_env)

    stats_path = os.path.join("results", f"{tag}_eval.json")
    _run([PYTHON, "main.py", "play", "--no-gui", "--agents", agent,
          "--scenario", scenario, "--n-rounds", str(eval_rounds),
          "--seed", str(eval_seed),
          "--continue-without-training",
          "--save-stats", stats_path, "--match-name", tag], run_env)

    full_path = os.path.join(REPO_ROOT, stats_path)
    with open(full_path) as f:
        data = json.load(f)
    data["_meta"] = {"board_seed_train": train_seed, "board_seed_eval": eval_seed,
                      "agent_seed": seed, "train_rounds": train_rounds,
                      "eval_rounds": eval_rounds}
    with open(full_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def summarize(data, agent, max_target):
    rounds = list(data["by_round"].values())
    n_rounds = len(rounds)
    completed = [r for r in rounds if r["coins"] >= max_target]
    completion_rate = len(completed) / n_rounds
    steps_given_completion = [r["steps"] for r in completed] if completed else []

    lifetime = data["by_agent"][agent]

    return {
        "completion_rate": completion_rate,
        "avg_coins": sum(r["coins"] for r in rounds) / n_rounds,
        "avg_steps_given_completion": (statistics.mean(steps_given_completion)
                                        if steps_given_completion else None),
        "avg_steps": sum(r["steps"] for r in rounds) / n_rounds,  # "survival steps"
        "suicide_rate": sum(r.get("suicides", 0) for r in rounds) / n_rounds,
        "crates_per_round": lifetime.get("crates", 0) / n_rounds,
        "bombs_per_round": lifetime.get("bombs", 0) / n_rounds,
        "invalid_per_round": lifetime.get("invalid", 0) / n_rounds,
        "n_rounds": n_rounds,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="qlearn_agent")
    p.add_argument("--scenario", required=True)
    p.add_argument("--max-target", type=int, required=True,
                    help="coins needed for a round to count as 'completed'")
    p.add_argument("--train-rounds", type=int, default=300)
    p.add_argument("--eval-rounds", type=int, default=100)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    env = os.environ.copy()
    per_seed = []
    for seed in range(args.seeds):
        tag = f"{args.tag}_seed{seed}"
        data = train_then_eval(args.agent, args.scenario, args.train_rounds,
                                args.eval_rounds, tag, seed, env)
        summary = summarize(data, args.agent, args.max_target)
        per_seed.append(summary)
        print(f"seed {seed}: completion_rate={summary['completion_rate']:.2f} "
              f"avg_coins={summary['avg_coins']:.2f} "
              f"steps|completed={summary['avg_steps_given_completion']} "
              f"avg_steps={summary['avg_steps']:.1f} "
              f"suicide_rate={summary['suicide_rate']:.2f} "
              f"crates/round={summary['crates_per_round']:.2f} "
              f"bombs/round={summary['bombs_per_round']:.2f}")

    def agg(key):
        vals = [s[key] for s in per_seed if s[key] is not None]
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.2f} +/- {std:.2f}"

    print()
    print(f"=== {args.tag} over {args.seeds} seeds "
          f"(train={args.train_rounds}, eval={args.eval_rounds}) ===")
    print(f"completion_rate:            {agg('completion_rate')}")
    print(f"avg_coins:                  {agg('avg_coins')}")
    print(f"avg_steps_given_completion: {agg('avg_steps_given_completion')}")
    print(f"avg_steps (survival):       {agg('avg_steps')}")
    print(f"suicide_rate:               {agg('suicide_rate')}")
    print(f"crates_per_round:           {agg('crates_per_round')}")
    print(f"bombs_per_round:            {agg('bombs_per_round')}")
    print(f"invalid_per_round:          {agg('invalid_per_round')}")


if __name__ == "__main__":
    main()
