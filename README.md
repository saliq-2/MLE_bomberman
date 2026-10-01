Please switch to the other branch

# bomberman_rl

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

---

## This fork: final project submission

Two learning agents, both linear function approximation over a shared
hand-crafted feature set, differing only in the temporal-difference target:

| Agent | Method | Files |
|---|---|---|
| `agent_code/qlearn_agent` | Off-policy Q-learning, semi-gradient TD(0), uniform experience replay, target network | `callbacks.py`, `train.py` |
| `agent_code/sarsa_agent` | On-policy SARSA — identical features, rewards and hyperparameters; only the TD target and the deferred-update bookkeeping differ | `callbacks.py`, `train.py` |

Three further directories exist only as experimental conditions and are **not**
submission candidates. They are kept because the numbers in Experiment 27
cannot be reproduced without them:

| Directory | Condition |
|---|---|
| `agent_code/sarsa_control` | 30 features, collinearity defect present — the Experiment 26 control |
| `agent_code/sarsa_control_fixed` | 30 features, defect fixed |
| `agent_code/sarsa_fixed` | 32 features, defect fixed — the winning cell of that factorial |
| `agent_code/sarsa_vs_rb3` | Experiment 28 stage 2, curriculum vs 3x `rule_based_agent` — produced the submitted checkpoint |
| `agent_code/sarsa_vs_t3` | Experiment 28 stage 2, curriculum vs `peaceful_agent` + `coin_collector_agent` — the Task 3 result |
| `agent_code/sarsa_threat_s0`, `_s1`, `_s2` | Experiment 29, opponent-threat features at three training seeds — a negative result, kept because it is the only measurement of training-seed variance in the project |
| `agent_code/sarsa_rb3_s1`, `_s2` | Experiment 28's condition at two further training seeds, to put an honest interval on the submitted checkpoint |

`callbacks.py` is byte-for-byte identical between `qlearn_agent` and
`sarsa_agent`, so any difference in their behaviour is attributable to the
update rule alone. That is deliberate: the comparison is a controlled one, with
the TD target as the single manipulated variable (see Experiment 21).

**Submitted checkpoint:** `agent_code/sarsa_agent/model.pt`, which is
`model_exp28_curriculum_rb3.pt` — a two-stage curriculum agent (solo Task 2,
then 1000 further rounds against three `rule_based_agent`s). The choice between
candidates is decided by measurement, not preference: see the factorial in
Experiment 27 and the tournament table in Experiment 28. Earlier candidates
ship alongside it so the choice is reversible without retraining.

**Curriculum training** is driven by two environment variables, both read only
when `self.train` is set, so evaluation and tournament play are unaffected:

```bash
QLEARN_INIT_FROM=agent_code/sarsa_agent/model_exp27_decollinearized.pt QLEARN_EPSILON_START=0.15 QLEARN_SEED=0 python main.py play --no-gui --agents sarsa_agent     rule_based_agent rule_based_agent rule_based_agent     --train 1 --scenario classic --n-rounds 1000 --seed 0     --continue-without-training
```

### Requirements

`numpy` only, beyond the framework's own dependencies — no additional packages
are needed to *run* the agent, so no `requirements.txt` is required for the
submission. `matplotlib` is used by the figure script, which is a reporting
tool and is never imported by agent code.

### Where things are

| Path | What it holds |
|---|---|
| `docs/experiments.md` | The full experiment log, 26 experiments. Starts with a section mapping the log onto the required report structure. |
| `results/` | Raw `--save-stats` JSON for every run cited in the log |
| `logs_by_tag/` | Per-run `game.log`, kept because win/draw/loss and the self-kill cause split are reconstructed from it |
| `figures/` | Report figures, all regenerated from `results/` |
| `scripts/` | Experiment harnesses and analysis tools (below) |

### Scripts

| Script | Purpose |
|---|---|
| `run_experiment.py` | Train a fresh model for N rounds, freeze it, evaluate with `train=False` over M rounds, repeat across seeds |
| `eval_vs_opponents.py` | Evaluate a checkpoint on any of the five configurations — `task1` (solo, coin-heaven), `task2` (solo, classic), `task3`, `task4`, `tournament` — with win/draw/loss reconstructed from `game.log` and cross-checked against `--save-stats`. Defaults to the three opponent configurations so older invocations reproduce exactly. |
| `summarize_runs.py` | Aggregate finished runs into one JSON of per-seed values and per-config means (no games re-run) |
| `make_report_figures.py` | Regenerate every report figure from `results/` |
| `pad_checkpoints.py` | Zero-pad older checkpoints to the current feature count, re-verifying numerical equivalence on each call |
| `check_escape_equivalence.py` | Assert the escape-redundancy search agrees with the original escape check wherever the two should agree |
| `audit_constant_features.py` | Flag features that never vary during solo training — a constantly non-zero one is collinear with the bias (Experiment 27). Worth running before adding any feature. |
| `classify_selfkills.py` | Split self-kills by whether an opponent bomb landed inside our own drop-to-death window |
| `eval_baseline_tournament.py`, `reprocess_exp24.py`, `bucket_selfkills.py`, `check_seed_variation.py` | Per-experiment analysis, kept so the numbers in the log stay reproducible |

### Reproducing the headline results

```bash
# Train a fresh checkpoint (classic board, single agent, 1000 rounds)
QLEARN_SEED=0 python main.py play --no-gui --agents sarsa_agent --train 1 \
    --scenario classic --n-rounds 1000 --seed 0 --continue-without-training

# Evaluate it against the provided agents, 200 rounds x 5 seeds, three configs
python scripts/eval_vs_opponents.py --agent sarsa_agent \
    --checkpoint agent_code/sarsa_agent/model.pt --seeds 5 --tag my_run

# Aggregate and plot
python scripts/summarize_runs.py --out results/summary_my_run.json \
    --run "My run:sarsa_agent:my_run"
python scripts/make_report_figures.py
```

Evaluation always runs with `self.train=False`, which sets ε to 0 — the same
condition the tournament uses. Training-time statistics are never reported as
performance numbers, since they are dominated by exploration.

**On the error bars in this project.** Up to Experiment 29 every experiment
trained a single model at seed 0 and reported the spread over evaluation
boards. Experiment 29 trained three seeds of one condition and got 0.400 /
0.747 / 1.002 score/round — roughly an order of magnitude more spread than the
board-level figures. Treat any ± in `docs/experiments.md` as evaluation noise
only, and any between-condition margin as directional unless that entry says
it was measured across training seeds.
