# Experiment Log

Internal working log of experiments, not the final report. Kept in the repo so
results are reproducible and dated; the actual PDF report (per project spec)
must not live in this repo.

Environment: conda env `ml_homework` (Python 3.11, numpy/scipy/scikit-learn/tqdm/pygame),
created per the project PDF's setup instructions:
`conda create -n ml_homework python=3.11 numpy scipy scikit-learn -y` then
`pip install pygame tqdm` inside it.

Gotcha found while setting this up: in this sandboxed shell, `conda activate
ml_homework` followed by `python ...` gets silently killed (exit 137) even for
a bare `print`. Invoking the env's interpreter by its direct path works fine:
`/usr/local/anaconda3/envs/ml_homework/bin/python3 main.py ...`. All commands
below use the direct path for that reason — outside this sandbox (e.g. a
normal terminal), plain `conda activate ml_homework && python main.py ...`
should work as documented in the project PDF.

**Reproducibility harness**: `scripts/run_experiment.py` trains a fresh model
for N rounds, freezes it, evaluates over M rounds with `self.train=False`
(so `epsilon=0`, i.e. the actual condition the tournament runs under), and
reports completion rate / steps-conditional-on-completion separately instead
of one blended "avg steps" figure. Repeats over several seeds (agent-side
randomness is *not* controlled by `--seed`, which only fixes the board layout
— see main.py/environment.py — so repeated invocations already vary) and
prints mean ± std. Added after review feedback (see below); earlier
experiments in this log predate it and are marked accordingly.

---

## Agent: `qlearn_agent`

Linear function-approximation Q-learning (semi-gradient TD(0) + uniform
experience replay + target network), one weight vector per action. This is
the "lecture technique" model (linear RL with hand-crafted features), per the
project's requirement that at least one model use lecture material.

**Actions**: `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`

**Features** (`state_to_features`, 30-dim as of Experiment 24 — grew from an
original 9-dim/5-action Task-1-only version through Experiments 1-18 (23
dims) and Experiment 24 (+7, opponent-aware), each addition documented
where it happened):
- bias (1)
- one-hot direction of the first step of a path to a *committed* target tile
  (nearest coin if any are visible, else nearest tile adjacent to a crate),
  suppressed to all-zero while in danger (Experiment 15) — 4 dims
- one-hot: which of the 4 directions is immediately walkable (4 dims)
- in-danger indicator (1 dim)
- one-hot: which of the 4 directions is both walkable *and* leads out of
  danger (4 dims)
- can-safely-drop-bomb-here indicator, via a time-expanded safety search
  (Experiment 13/14) — 1 dim
- crate-payoff: crates a bomb dropped here would destroy, capped/scaled
  (Experiment 11) — 1 dim
- crate-distance: BFS steps to the nearest crate-adjacent tile (Experiment
  11) — 1 dim
- `crate_payoff * safe_bomb` interaction term (Experiment 12) — 1 dim
- on-target indicator, disambiguating "no target" from "standing on target"
  (Experiment 13) — 1 dim
- committed escape-direction one-hot, same commitment pattern as the target
  direction but for escaping danger (Experiment 18) — 4 dims
- BFS distance to a tile adjacent to the *committed* opponent target,
  capped/scaled (Experiment 24) — 1 dim
- one-hot direction of the first step toward that committed opponent,
  suppressed to all-zero while in danger, same commitment pattern as the
  coin/crate target (Experiment 24) — 4 dims
- opponent-in-blast-range-if-I-bomb-now indicator, reusing the crate-payoff
  blast computation (Experiment 24) — 1 dim
- opponent-is-trapped indicator: would the committed opponent have no
  escape if I bombed now, via the same time-expanded safety search as
  can-safely-drop-bomb-here (Experiment 24) — 1 dim

(No separate "bomb available" feature — removed in Experiment 10, collinear
with safe-to-bomb whenever bombing is viable.)

**Opponent targeting** (Experiment 24): same committed-target pattern as
coins/crates — commit to the nearest opponent by name and hold that
commitment across steps (invalidated only on reach or elimination), rather
than re-deriving "nearest" every step, since that oscillation was the exact
failure diagnosed in Experiments 1c and 13. The BFS goal is "adjacent to the
opponent," not "on the opponent's tile" — the opponent's own tile is in the
`occupied` set, so a naive same-tile goal is filtered out by the walkability
check before the goal predicate ever runs and silently returns
"unreachable" every time (caught via synthetic testing before any training
run, not through a live game).

**Hyperparameters** (current, after Experiment 8): `alpha=0.01, gamma=0.9`
(overridable via `QLEARN_GAMMA`, Experiment 17), `epsilon: 0.3 -> 0.05
(x0.995/round), replay batch=8/step (min buffer 200, capacity 10k), target
network synced every 500 gradient updates`. (Was `alpha=0.1`, no replay, no
target network, for Experiments 1-5 — see Experiments 6-8 for why that
changed.)

**Rewards** (default, shaping enabled — see `train.py` for the full table;
current as of Experiment 24): sparse/base always apply —
`COIN_COLLECTED +1`, `COIN_FOUND +0.2`, `CRATE_DESTROYED 0` (moved to
`CRATE_PAYOFF_AT_DROP +0.3`/crate, paid at drop time not destruction —
Experiment 16), `KILLED_SELF 0` (was -5, removed — fires alongside
`GOT_KILLED` on every self-kill, was double-charging -10 — Experiment 16),
`GOT_KILLED -5`, `KILLED_OPPONENT +2.0` (Experiment 24 — real game weights a
kill 5x a coin; 2x reflects that without letting one sparse event dominate
the reward scale), `SURVIVED_ROUND +1`. Shaping terms (stripped by
`QLEARN_USE_SHAPING=0`): `INVALID_ACTION -1`, `WAITED -0.1`,
target-direction ±0.1, opponent-direction ±0.1 (Experiment 24, symmetric
with the coin/crate target shaping), bomb-drop-safety ±0.2/-0.5, plus
potential-based escape shaping `Φ(s')-Φ(s)` where `Φ(s) = -0.5 × (BFS
distance to nearest safe tile)`, 0 when not in danger (Experiment 15,
replaced the earlier flat ±0.3 danger events).

---

### Experiment 1 — Task 1 baseline, coin collection with reward shaping

*(Original run, 9-feature/5-action agent, before Task 2 or the eval harness
existed. Superseded by Experiment 1c below, which found and fixed real bugs
this run's methodology couldn't see. Kept for the record.)*

**Setup**: 300 rounds, `coin-heaven`, stats measured *during* training
(`self.train=True` throughout, ε decaying from 0.3).

**Results**: 48.0/50 coins/round average, 265/300 rounds (88%) with all 50
coins, 3052 invalid actions/300 rounds. No visible learning curve — near
ceiling from round 1.

**Caveat (added after review)**: this measures training-time behavior with
residual ε-exploration, not the actual deployed (ε=0) policy — which turned
out to matter a lot; see Experiment 1c.

---

### Experiment 2 — Ablation: sparse reward only (no shaping)

*(Same vintage/caveat as Experiment 1.)*

**Setup**: identical to Experiment 1, `QLEARN_USE_SHAPING=0`.

**Results**: 36.1/50 coins/round, 99/300 rounds (33%) with all 50 coins,
19,284 invalid actions/300 rounds (vs 3,052 with shaping — 6.3x more).

**Conclusion (still holds directionally)**: shaping matters — outperforms
the sparse-only ablation by +33% coins and 16,232 fewer invalid actions
over 300 rounds. Re-run under the fixed hyperparameters and a real greedy
eval is Experiment 9 below, which confirms the direction but with a smaller,
noisier gap than this (uncorrected) comparison suggested.

---

### Experiment 3 — Task 2 baseline, crates + bombing, no opponents

**Setup**: `classic` scenario (17x17, 0.75 crate density, 9 coins), single
agent, 500 rounds, old hyperparameters (`alpha=0.1`, no replay/target net).
`results/task2_run1.json`

**Results** (blocks of 50 rounds): avg coins climbs unevenly (0.06 → 0.16 /
9), avg steps-to-death/completion climbs steadily (56.8 → ~200), but
**suicide rate stays extremely high throughout: 91.6% overall**, only
trending from ~100% (rounds 1-50) to ~76-84% (rounds 300-450) before
bouncing back to 100% in the last block. Crates destroyed: 2138/500 rounds
(~4.3/round) — the agent is actively engaging with bombing, just dying to it
almost every time.

**Read**: partial learning (survival time increasing) but bomb-safety is
clearly the dominant bottleneck, exactly as the PDF's task description
warns ("escaping bombs is a crucial capability... place proper emphasis on
this step").

---

### Experiment 4 — Safety-margin fix for the escape check

**Hypothesis**: the escape-BFS (`_can_escape_own_bomb`) uses the full
`BOMB_TIMER` window with no buffer, so one wasted step (a bumped wall, an
exploration move) turns a "safe" bomb into a fatal one. Added a 1-step
margin (`BOMB_TIMER - 1`).

**Setup**: identical to Experiment 3 (500 rounds, `classic`, old
hyperparameters). `results/task2_run2_margin.json`

**Results**: essentially unchanged — 93.0% suicide rate overall (vs 91.6%
before), within noise. **The margin wasn't the bottleneck.**

---

### Experiment 5 — Does more training data alone fix it?

**Setup**: same as Experiment 3/4, but 3000 rounds instead of 500.
`results/task2_run3_longer.json`

**Results**: suicide rate is **flat at 85-88% from round 300 all the way to
round 3000** (10 blocks of 300 rounds, min 85.0%, max 87.7%) — no
improvement with 6x more data.

**Conclusion**: this is a genuine plateau, not a data-volume problem. Online
single-sample TD(0) has a credit-assignment problem here: the terminal -10
penalty (`KILLED_SELF + GOT_KILLED`) only updates the Q-value of whatever
action was taken on the *very last* step before death — not the actual
bomb-drop decision 3-4 steps earlier that caused it. Propagating that signal
backward through a multi-step chain via single-step bootstrapping, from a
single trajectory, is slow-to-nonexistent without experience replay. This
motivated Experiments 6-8.

**Caveat**: this diagnosis was never actually tested against the alternative
explanation — it's a plausible mechanism, argued from first principles, not
a verified cause. Experiments 6-8 were built and validated entirely on Task 1
(which has no credit-assignment problem of this kind — coin-collection reward
is immediate, not delayed several steps behind a terminal penalty). Whether
replay + target network actually fixes *this* plateau, rather than some
other bug specific to Task 2's bombing logic, is exactly what Experiment 10
below tests.

---

## Review feedback and methodology fixes

External review of this log (reasoning from the numbers as written, before
any of Experiments 6-9 existed) flagged, in order of cost:

1. **Arithmetic error** in Experiment 2's conclusion (fixed above).
2. **Training-time metrics ≠ deployment metrics.** Experiments 1/2/3/4/5 all
   measure `self.train=True` behavior with residual ε-exploration. The
   tournament runs `self.train=False`, i.e. pure greedy. Needed a real
   train/eval split — built as `scripts/run_experiment.py`.
3. **"avg steps" conflates two things**: genuine completion time vs. hitting
   the step cap on a failed round. Needed separate completion-rate and
   steps-conditional-on-completion metrics (now in the harness).
4. **Single runs, no variance** — block-to-block spread in Experiment 1
   (45.3 to 49.9) is noise with no way to say so. Needed multi-seed mean±std
   (now in the harness; agent-side randomness isn't fixed by `--seed`, so
   repeated invocations already differ).
5. **Task 1's near-ceiling result risks reading as a deterministic-policy
   violation** (project rule: no feature that just returns the best move).
   Framing: the BFS-direction feature makes Task 1 *easy*, not solved by
   fiat — the model still has to learn to trust and act on it, and Task 2
   (below) is where that stops being sufficient. Worth stating explicitly in
   the report rather than leaving for a tutor to raise.

Applying fix #2 to Task 1 (below) surfaced two real, previously-invisible
bugs — proof the methodology issue was substantive, not cosmetic.

---

### Experiment 1c — Proper eval reveals an oscillation trap

**Setup**: `scripts/run_experiment.py`, `qlearn_agent` (now the 16-feature/
6-action Task-2-capable version — see note below), `coin-heaven`, train=300,
eval=100, 5 seeds.

**Result**: **completion_rate=0.00 across all 5 seeds**, avg coins 1.4-9.5/50
— a severe regression from Experiment 1's reported 48/50. Game log showed
why: the eval (pure greedy) policy gets stuck alternating `UP, DOWN, UP,
DOWN, ...` for the rest of the round (confirmed systemic: 14,987 `UP` vs
14,931 `DOWN` across the log, near-perfectly balanced, not a rare edge case).

**Root cause**: `state_to_features` picked "direction to nearest coin" fresh
every step. Near a position equidistant between two coins, the identity of
"nearest" flips as the agent moves one step closer to either — inducing a
stable 2-cycle under a deterministic greedy policy. Training-time ε=0.05
floor had been just enough noise to occasionally break this tie, hiding the
bug in Experiments 1/2's training-time metrics entirely.

**Fix**: commit to one target tile (`self.current_target`, persisted across
`act()` calls) and path to that *specific point* via point-to-point BFS
(`_bfs_to_point`) until it's collected or invalidated, instead of re-picking
"nearest of many" every step. Verified in isolation with a symmetric
two-coin repro (monotonic single-direction approach, no oscillation).
Target is reset at the start of each round (`step == 1`, checked in `act()`
itself so it works in eval mode too, where `train.py`'s `end_of_round` never
runs).

**Note on the regression's size**: part of the 48→~1 drop is a confound, not
just the oscillation bug — this eval used the current 16-feature/6-action
agent (built for Task 2) on the Task 1 scenario, with only 300 rounds'
training budget, which is undertrained for the larger model regardless (a
follow-up 800-round/3-seed check still only reached 0.01 completion rate
with high seed variance, 2.0-31.3 avg coins). The oscillation bug is real and
confirmed independently either way (see the UP/DOWN log evidence above).

---

### Experiment 6 — Added experience replay (motivated by Experiment 5's plateau)

**Change**: `_remember_and_replay` — store every transition in a deque,
uniform-sample a batch of 32 and apply the same TD update to each, every
step, alongside the existing single-sample online update. Direct response to
Experiment 5's diagnosis (online TD(0) can't propagate a rare, high-magnitude
terminal penalty back through a multi-step chain from a single trajectory).

**Result on Task 1** (`coin-heaven`, same eval harness): **worse**, not
better — avg coins dropped to 1.17 ± 1.10/50 (vs ~3/50 without replay after
the target-commit fix). Action log: 35,119/40,000 actions were `WAIT` — a
new, worse absorbing trap.

**Diagnosis**: `WAIT` is a literal self-loop transition — the feature vector
is identical before and after (the agent's position doesn't change). With no
target network, bootstrapping `max_a' Q(s', a')` off the *same* weights being
updated means every replay of that self-loop bootstraps off an
already-updated estimate of itself. Resampled repeatedly and uniformly by
replay (same weight as any other transition), this compounds into a runaway
feedback loop — a known instability (the "deadly triad": function
approximation + bootstrapping + this kind of repeated off-policy-like
resampling) that a target network exists specifically to prevent.

---

### Experiment 7 — Added a target network (round-synced)

**Change**: `self.target_weights`, a copy of the weights used only for
computing bootstrap targets, synced to the live weights every 5 rounds.

**Result on Task 1**: partial improvement (pure-WAIT collapse stopped) but a
**new** oscillation appeared — `RIGHT`/`LEFT` this time (13,673 vs 13,523
across the log, plus 12,762 `WAIT`). avg coins barely moved: 1.09 ± 0.53/50.

**Diagnosis, confirmed by dumping the weight matrix** after a clean 300-round
run: every action's bias weight was large and positive (4.8-7.2, `WAIT`
highest at 7.19), and the four target-direction weights were positive across
*every* action row instead of forming a clean diagonal (action X strongly
weighted on "target says X", weakly/negatively on the others). That's
divergence, not undertraining. Cause: at `alpha=0.1` with 33 gradient updates
per environment step (1 online + 32 replay), the effective step size is far
too aggressive, and syncing the target network every 5 *rounds* — up to tens
of thousands of updates at this update rate — is nowhere near tight enough
to counteract it. (A concurrent background run that raced on the same
`model.pt` file was caught and killed before being mistaken for real data —
the divergent weight pattern was independently reproduced on a clean,
sequential run afterward.)

---

### Experiment 8 — Fix: lower effective learning rate, update-count target sync

**Changes**:
- `ALPHA`: 0.1 → 0.01
- `REPLAY_BATCH_SIZE`: 32 → 8 (9 updates/step instead of 33)
- Target network synced every 500 gradient *updates* (tracked via a counter
  incremented inside `_update`), not every N rounds

**Weight matrix after a clean 300-round run**: bias weights now 0.06-2.5
(down from 4.8-7.2), and target-direction weights show the expected diagonal
— e.g. `tgt_UP` is highest in its column for the `UP` row (0.83), `tgt_RIGHT`
highest for `RIGHT` (0.53), `tgt_DOWN` highest for `DOWN` (0.74), `tgt_LEFT`
highest for `LEFT` (0.79).

**Eval results** (`coin-heaven`, train=300, eval=100, 5 seeds):

| seed | completion rate | avg coins/50 | steps\|completed |
|---|---|---|---|
| 0 | 1.00 | 50.0 | 125.1 |
| 1 | 1.00 | 50.0 | 124.6 |
| 2 | 0.00 | 7.5 | — |
| 3 | 1.00 | 50.0 | 125.9 |
| 4 | 1.00 | 50.0 | 124.2 |

**Aggregate**: completion rate 0.80 ± 0.45, avg coins 41.5 ± 19.0/50, steps
given completion 124.9 ± 0.75 (tight — the 4 successful seeds converge to
almost identical solve times).

**Read**: 4/5 seeds now solve Task 1 cleanly and consistently under a true
greedy eval — first time this agent has actually demonstrated the ~48-50/50
performance Experiment 1's training-time metrics *implied* but never
verified. One seed (2/5) still finds a degenerate policy — some residual
seed-sensitivity remains, worth reporting honestly rather than cherry-picking
a good seed. Not yet re-validated on Task 2 (Experiment 3-5's plateau was
measured under the old, unstable hyperparameters — that plateau number is
now suspect and needs re-running under this fixed config; see Next steps).

---

### Experiment 9 — No-shaping ablation, re-run under fixed hyperparameters

**Setup**: identical protocol to Experiment 8 (`coin-heaven`, train=300,
eval=100, 5 seeds) but `QLEARN_USE_SHAPING=0` — supersedes Experiment 2 as
the methodologically-correct version of this comparison.

| seed | completion rate | avg coins/50 | steps\|completed |
|---|---|---|---|
| 0 | 0.00 | 6.6 | — |
| 1 | 0.11 | 29.3 | 126.3 |
| 2 | 0.00 | 6.8 | — |
| 3 | 1.00 | 50.0 | 126.0 |
| 4 | 0.00 | 1.0 | — |

**Aggregate**: completion rate 0.22 ± 0.44, avg coins 18.75 ± 20.57/50 (vs
Experiment 8's shaping-on aggregate: 0.80 ± 0.45 completion, 41.5 ± 19.0/50
coins).

**Conclusion**: with the instability fixed and a real greedy eval, shaping
still clearly wins — 0.80 vs 0.22 completion rate, roughly +23 coins/round on
average — but the gap is smaller and noisier than Experiment 2's original
(uncorrected) 48.0 vs 36.1 suggested, and both arms now show high seed
variance (std ≈ mean, in the no-shaping case). The honest conclusion is
"shaping helps, and seed variance is large enough on both arms that more
seeds would sharpen this" rather than a clean, low-variance win — a materially
different and more defensible claim than Experiment 2's, and a direct result
of the review feedback (methodology point #4).

---

### Experiment 10 — Task 2 re-run under the Experiment 8 hyperparameters, properly seeded

**Setup**: `scripts/run_experiment.py --scenario classic --max-target 9
--train-rounds 1000 --eval-rounds 200 --seeds 5`, both board seed (disjoint
train/eval ranges) and `QLEARN_SEED` (agent RNG, newly added — see below)
set explicitly per repeat, recorded in each result's `_meta`.

**Result**: identical across all 5 seeds — **0 coins, 0 crates destroyed, 0
bombs dropped, 0 suicides, every round runs the full 400-step cap.** Not the
Experiment 3-5 self-kill plateau at all: the opposite failure. Action log
(seed 4): 22,808 `LEFT` / 22,790 `RIGHT` / 17,202 `UP` / 17,202 `DOWN`, no
`BOMB`, no `WAIT` — the agent wanders indefinitely and never engages the
core mechanic.

**Diagnosis #1 (confirmed, fixed)**: dumped the weight matrix. `BOMB`'s
`bomb_avail` weight was -3.19 against a `safe_bomb` weight of only +0.92 —
net `Q(BOMB)` strongly negative *even when safe to bomb*. Root cause:
`safe_bomb` (feature 14) can only be 1 when `bomb_avail` (feature 15) is
also 1 — they're perfectly collinear whenever bombing is actually viable —
so the model had no clean way to attribute "bombing is fine here" to one
feature without the other absorbing an offsetting weight. **Fix applied**:
removed the redundant `bomb_avail` feature entirely (`INVALID_ACTION`
already penalizes trying to bomb with none left, so nothing is lost).
`N_FEATURES`: 16 → 15.

**Diagnosis #2 (confirmed, not yet fixed)**: re-trained and re-evaluated
(seed 0) after the fix above — **still 0 bombs, 0 score.** The negativity
just moved: `BOMB`'s bias dropped to -4.34 (from -1.39), `safe_bomb` stayed
around +0.69, net still clearly negative in typical states. This isn't the
same bug recurring, it's a deeper one: with **uniform** experience replay,
early training (near-zero weights, ε starting at 0.3) generates bombing
attempts in largely uninformed, frequently-unsafe positions, well before the
model has learned enough to place bombs well. Bad outcomes vastly
outnumber good ones in that early data, and uniform sampling from the
replay buffer just reinforces that skewed empirical average — there's no
mechanism to up-weight the sparse, valuable "safe bombing near a crate
worked out" transitions relative to the abundant "random bombing was bad"
ones. This is a second, independent line of evidence (distinct from
Experiment 5's credit-assignment argument) for the same fix: **prioritized
replay**, not uniform, is very likely required to make bombing learnable
here at all — not just to learn to escape it safely.

**Read**: the Experiment 8 stability fix generalizes (no divergence, no
oscillation, clean convergence, zero suicides) but over-corrected into a
different degenerate optimum specific to Task 2. The original Experiment
3-5 suicide-plateau question is now moot for this config — the agent no
longer suicides because it no longer bombs at all, which is arguably worse
for the actual task. Task 2 is not solved by any experiment in this log yet.

---

### Experiment 11 — Diagnostic + crate-payoff/crate-distance features

**Diagnostic, before changing anything**: built a real `classic` board and
drove the agent's own `state_to_features`/`_select_target` step by step
(not through the full game loop) to directly inspect what happens at the
moment the agent reaches its committed crate-adjacent target. Confirmed:
the target-direction one-hot correctly goes all-zero exactly when standing
on the target, and `current_target` does **not** get re-picked — it stays
fixed across 6+ consecutive steps sitting there (validity check only
re-fires if the target stops being crate-adjacent, e.g. once the crate is
destroyed). **No 2-cycle from target reselection.** Experiment 10's
"wanders forever, never bombs" behavior is confirmed to be purely a
Q-value problem, not a target-selection bug.

**Change**: added two features (`N_FEATURES`: 15 → 17):
- `crate_payoff`: crates a bomb dropped on the current tile would destroy
  (via `get_blast_coords`), capped at 4 and scaled to [0,1].
- `crate_dist`: BFS distance to the nearest crate-adjacent tile, capped at
  10 and scaled to [0,1].

Sanity-checked on a real board before training: at the start position,
`crate_payoff=0.75` (3/4 crates in range) and `crate_dist=0.10`; after
moving onto the crate-adjacent target, `crate_dist=0.00` as expected.

**Setup**: Experiment 10's exact protocol, unchanged — `classic`,
train=1000, eval=200, 5 seeds, board+agent seeds controlled.

**Result**: identical to Experiment 10 across all 5 seeds — **0 bombs/round,
0 crates/round, 0 coins, 0 suicides, every round hits the 400-step cap.**
The new features did not fix it.

**Weight matrix, `BOMB` row** (seed 0, after 1000 rounds):

| bias | crate_payoff | crate_dist | safe_bomb | danger |
|---|---|---|---|---|
| -4.784 | **-0.585** | 0.035 | 0.372 | -2.351 |

**Diagnosis**: `crate_payoff`'s weight is *negative* — the model learned
that more crates in blast range makes `BOMB` *worse*, the opposite of the
intended signal. This is explainable, not noise: crate-dense tiles are also
the tiles with the fewest escape routes, so in the training data,
`crate_payoff` and "unsafe to bomb" are positively correlated — during
early random exploration, high-payoff bombing spots disproportionately
co-occur with self-kills (the same population-level skew diagnosed in
Experiment 10's diagnosis #2, now visible in a second feature). A purely
additive linear model (`Q = w · phi(s)`, one weight per feature, no
interactions) cannot represent "good if high-payoff **and** safe, bad if
high-payoff **but** unsafe" — that's a conjunction of two features, and
gradient descent on a confounded data distribution finds the marginal
(wrong) association instead of the conditional (right) one. Adding another
independent additive feature couldn't have fixed this in principle; the
model needed either an explicit interaction term (e.g. `crate_payoff *
safe_bomb`, or `crate_payoff` gated to 0 when `safe_bomb=0`) or a training
distribution where safe and unsafe high-payoff states are represented in
closer to their true (not exploration-skewed) proportions — which is,
again, what prioritized replay would do (up-weighting the rare "safe AND
high-payoff" transitions relative to the abundant "unsafe AND high-payoff"
ones), converging with Experiment 10's diagnosis #2 from a different angle.

**Read**: two independent lines of evidence now point at the same two
fixes, and neither has been tried yet: (a) an interaction feature between
payoff and safety, cheap to add and test in isolation; (b) prioritized
replay, the larger change. (a) is a five-minute change and should be tried
first, since it might resolve this without touching the training loop at
all.

---

### Experiment 12 — Interaction feature (`crate_payoff * safe_bomb`)

**Change**: added `payoff_x_safe = crate_payoff * safe_bomb` as its own
dimension, kept bare `crate_payoff` alongside it. `N_FEATURES`: 17 → 18.

**Result**: identical to Experiment 10/11 — 0 bombs/round, 0 crates, 0
coins, 0 suicides, all 5 seeds. The interaction term alone did not unlock
bombing at eval, though (see Experiment 18) it later turned out to matter
once the deeper representation issue was also fixed.

---

### Experiment 13 — On-target oscillation fix, ground-truth escape-check rewrite

Two independent investigations, run together:

**(1) On-target fix**: `_select_target` now invalidates the committed
target the instant `pos == target` (forcing fresh reselection next call),
and a new `on_target` feature (computed from the *prior* commitment, before
invalidation) disambiguates "no target" from "standing on target" — both
previously produced the same all-zero `target_onehot`. `N_FEATURES`: 18 → 19.

Re-run under the Experiment 10 protocol: **unchanged** — still 0
bombs/round across all 5 seeds. A fresh action-sequence dump showed *why*:
not the old 2-cycle, but a new 6-step commute between two different
crate-adjacent targets ((2,15) and (1,13)), `on_target=1` firing correctly
at each arrival, yet the agent still traveled back and forth instead of
bombing either one. The representation bug was fixed; the underlying
Q-value preference for movement over bombing was not.

**(2) `_can_escape_own_bomb` rewritten for correctness**: replaced the
static single-bomb BFS with a time-expanded (position, time) BFS accounting
for per-step danger timing, other active bombs, and the explosion residue
round — verified against 6 synthetic test cases (including a bug caught
and fixed during testing: an over-extended safety horizon that let
far-future bombs block near-term decisions).

Ground-truth check (1000 rounds, seed 0, `BOMB_TIMER+2`=6-step window):
`safe_bomb=1` → death within window: **46.41%** (907 drops, 530 deaths) vs
41.46% baseline — the rewrite did not reduce the false-positive rate, and
if anything it went up slightly. Ruled out "the escape-check's
spatial/temporal logic is buggy" as the explanation for the high
false-positive rate.

---

### Experiment 14 — (superseded numbering; see Experiment 13's escape-check rewrite above)

*(Kept as a placeholder: the escape-check rewrite is documented under
Experiment 13 to keep the two related investigations from that session
together. No separate content here.)*

---

### Experiment 15 — Potential-based escape shaping + target-pull suppression in danger

**Changes**:
- Removed the flat `MOVED_INTO_DANGER`/`MOVED_OUT_OF_DANGER`/`STAYED_IN_DANGER`
  events (±0.3). Replaced with potential-based shaping: `Φ(s) = -0.5 ×
  (BFS distance to nearest safe tile under the current bomb config)`, `Φ(s)
  = 0` when not in danger, `Φ(s_terminal) = 0` by convention. `Φ(s') - Φ(s)`
  added directly to the reward every transition (gated by `QLEARN_USE_SHAPING`,
  consistent with the other shaping terms).
- Zeroed `target_onehot` (indices 1-4) whenever `danger=1`, so escape/safety
  features are the only directional signal while a bomb is ticking — target
  pursuit resumes automatically once danger clears (`_add_shaping_events`
  needed no change: its existing `target_dirs.sum()>0` guard already
  handles an all-zero one-hot correctly, whether from "no target" or "in
  danger").

**Result** (seed 0): still 0 bombs, 0 crates, 0 coins, 0 suicides at eval.
Not yet fixed, but the reward mechanism was now principled rather than an
ad-hoc flat bonus, setting up Experiment 16.

---

### Experiment 16 — Remove double death penalty + pay crate reward at drop time

Two more reward-structure changes, requested together (reduced attribution
accepted given time constraints):

**(a) Double death penalty removed.** Verified in `environment.py` first:
`KILLED_SELF` (in the "Kill agents" loop) and `GOT_KILLED` (unconditionally,
in the following "Remove hit agents" loop, for every agent hit including
self-kills) both fire on *every* self-kill — so a self-kill had been
charged -5 + -5 = -10 since Experiment 3. `KILLED_SELF: 0.0` (was -5.0),
`GOT_KILLED` alone now carries -5.0.

**(b) Crate reward moved to drop time.** `CRATE_DESTROYED: 0.0` (was 0.3,
paid ~4 steps later at actual destruction, only if the agent survived to
see it). New `CRATE_PAYOFF_AT_DROP` event, appended once per crate the
just-dropped bomb will destroy (reusing the same blast-radius computation
as the `crate_payoff` feature), worth 0.3 each — paid immediately.

**Training result** (seed 0, 1000 rounds): suicide rate 82.6% (826/1000,
vs 86.3% baseline) — modest improvement. bombs/round 1.678 (vs 1.560
baseline), crates/round 4.658 (vs 3.904) — both up, real engagement
increase. Weight matrix: `BOMB`'s `payoff_x_safe` strengthened to +1.197
(from +0.660), bare `crate_payoff`'s confound shrank to -0.068 (from
-0.802) — meaningful structural improvement.

**Eval result** (seed 0): **still 0 bombs, 0 crates, 0 coins, 0 suicides.**
Training engagement improved substantially; eval (pure greedy) stayed at
exactly zero. This is where the epsilon-sensitivity sweep came in.

**Epsilon-sensitivity sweep** (same model, eval-only, varying ε):

| ε | bombs/round | crates/round | coins/round | suicide rate | avg steps |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.0% | 400.0 |
| 0.02 | 0.785 | 2.200 | 0.030 | 55.5% | 279.8 |
| 0.05 | 1.290 | 3.660 | 0.050 | 84.0% | 160.2 |
| 0.10 | 1.460 | 4.160 | 0.085 | 96.5% | 91.4 |

**Read**: the greedy "never bomb" policy is not a calibrated risk-averse
preference — it's a knife-edge equilibrium. The instant any exploration
noise lets bombing happen, engagement surges toward training-time levels
but the suicide rate rockets to 55-96%. Zero suicides at ε=0 was a trivial
consequence of never acting, not evidence of safety. The bottleneck isn't
"deciding to bomb" — it's what happens in the several steps *after* a bomb
is placed, which the greedy policy (whenever it's reached at all) executes
very badly.

---

### Experiment 17 — Discount factor (γ) sweep

Motivated by an independent prior team's report (Ernst & Striebel 2021,
"maverick" — a full past submission for this same course, read for
strategy only, no code touched): they found their own "shivering"/loop bug
needed γ reduced from 0.85→0.6 to fix, reasoning that weak short-horizon
features can't support a Q-function trying to approximate long-horizon
value, and a high γ makes "do nothing" locally optimal.

**Change**: `GAMMA` made overridable via `QLEARN_GAMMA` (default 0.9,
unchanged unless set) — an isolated, single-variable change.

**Result** (seed 0, γ=0.6): training suicide rate 84.0% (vs 82.6% at
γ=0.9), bombs/round 1.594 (vs 1.678), crates/round 4.602 (vs 4.658) — all
within noise, no real difference. Eval: still 0 bombs, 0 crates, 0 coins,
400 steps every round. `BOMB` row's `crate_payoff` improved marginally
(-0.019, near-zero) but `payoff_x_safe` and `bias` were comparable to
γ=0.9, and eval behavior was unchanged.

**Conclusion**: γ was not the lever. The prior team's finding was real for
their setup but didn't transfer — plausibly because their loop was a
value-estimation mismatch from much weaker features, not the specific
multi-target navigation cycle diagnosed here.

---

### Experiment 18 — Escape-commitment mechanism (breakthrough)

**Motivation**: Experiment 16's epsilon sweep showed the real bottleneck is
escape *execution*, not the drop decision. Structurally, this was the same
gap that `current_target`/`on_target` closed for navigation (Experiment
11/13): before that fix, the agent re-derived "which direction to go"
fresh every step with no commitment, causing oscillation. Escape had the
identical shape — `IDX_SAFE_BASE` could light up multiple simultaneously-
safe directions with no single best answer, and nothing committed the
agent to one, so it could still flicker mid-escape.

**Change**: added `self.escape_direction`, committed the same way as
`current_target` — computed once via the same time-expanded safety search
as `_can_escape_own_bomb`/`_distance_to_safety` (`_escape_first_step`), and
held (`_direction_still_safe` checks it's still valid each step) until it
stops being safe or danger clears, instead of being re-derived ambiguously
every step. New `escape_onehot` feature (4 dims, `IDX_ESCAPE_BASE`),
`N_FEATURES`: 19 → 23. Verified in isolation: deterministic, sticky, no
flip-flop, correctly recognizes "already safe" (returns `None`).

**Training result** (seed 0, 1000 rounds): bombs/round **4.617** (vs 1.678
in Experiment 16), crates/round **16.061** (vs 4.658), coins/round **0.915**
(vs 0.145) — all up 3-6x. Suicide rate 90.3% (vs 82.6%) — worse in
isolation, but alongside dramatically more attempted engagement.

**Eval result — first non-zero result in this entire investigation**
(seed 0): bombs/round 1.735, crates/round 6.420, coins/round 0.240,
suicide rate 23.0%, avg steps 310.1 (rounds now vary in length instead of
hitting exactly 400 every time).

**Weight matrix, `BOMB` row** (seed 0): `bias=-0.757` (least negative seen
in any experiment; range across Experiments 10-17 was -1.4 to -4.8),
`safe_bomb=+1.244` and `payoff_x_safe=+1.491` (strongest positive signals
yet), `crate_payoff≈-0.004` (no confound). Net Q(BOMB) in a safe,
high-payoff state ≈ -0.757+1.244+1.491 ≈ **+1.98**, clearly positive for
the first time.

**Full 5-seed eval** (`classic`, train=1000, eval=200, board+agent seeds
controlled):

| seed | bombs/round | crates/round | coins/round | suicide rate | avg steps |
|---|---|---|---|---|---|
| 0 | 1.735 | 6.420 | 0.240 | 23.0% | 310.1 |
| 1 | 2.860 | 10.350 | 0.415 | 28.0% | 292.9 |
| 2 | 1.770 | 5.975 | 0.275 | 11.0% | 359.0 |
| 3 | 1.880 | 7.520 | 0.240 | 4.0% | 384.5 |
| 4 | 1.510 | 5.170 | 0.120 | 28.0% | 294.6 |

**Aggregate**: bombs/round 1.95 ± 0.52, crates/round 7.09 ± 2.01, coins/round
0.26 ± 0.11, suicide rate 19.0% ± 11.0%, avg steps 328.25 ± 41.28,
invalid/round 0.07 ± 0.08 (very low).

**Read**: real, reproducible, holds across all 5 seeds — not a fluke.
Suicide rate is still high and seed-variable (4-28%), and this is
dramatically better than the epsilon-sweep's 55-96% (pure exploration-
triggered bombing with no commitment mechanism), confirming the escape
feature is doing real work, not just enabling recklessness. Task 2 is not
solved, but for the first time this agent engages with the core mechanic
at eval at all. This is the first genuine breakthrough after Experiments
10-17 all returned exactly zero across every metric.

---

### Experiment 19 — Does more training data narrow the seed variance?

**Setup**: identical to Experiment 18's protocol, but `train-rounds=5000`
instead of 1000 (~17 min/seed vs ~1 min/seed). No code changes.

**Full 5-seed eval**:

| seed | bombs/round | crates/round | coins/round | suicide rate | avg steps |
|---|---|---|---|---|---|
| 0 | 3.19 | 10.84 | 0.43 | 18.0% | 330.8 |
| 1 | 1.71 | 5.57 | 0.19 | 37.5% | 254.5 |
| 2 | 1.74 | 7.38 | 0.39 | 3.5% | 386.8 |
| 3 | 2.77 | 9.53 | 0.34 | 19.0% | 326.2 |
| 4 | 2.10 | 7.99 | 0.39 | 8.5% | 365.2 |

**Aggregate**: bombs/round 2.30±0.66 (vs Exp 18: 1.95±0.52), crates/round
8.26±2.02 (vs 7.09±2.01), coins/round 0.35±0.09 (vs 0.26±0.11), suicide
rate 17.0%±13.0% (vs 19.0%±11.0%), avg steps 332.72±50.36 (vs
328.25±41.28).

`BOMB` row (seed 0, 5000 rounds): `bias=-0.839, safe_bomb=+1.233,
crate_payoff=+0.171, crate_dist=+0.286, payoff_x_safe=+1.075` —
`crate_payoff` is now genuinely *positive* standalone (was ≈-0.004 at
1000 rounds), the confound fully resolved.

**Read**: modest mean improvement (~15-20% more coins/bombs) but the
seed-to-seed suicide-rate spread did **not** narrow (3.5-37.5% vs 4-28% at
1000 rounds — about the same width, shifted). 5x the training data bought
incremental engagement, not convergence to low variance — this looks like
a genuine ceiling for the current feature set/reward structure rather than
an undertraining artifact. 1000 rounds is the better default given the
~17x wall-clock cost for this modest gain, unless a later change shifts
that calculus. **Resolves Next Step 1 below** (more data doesn't fix the
variance).

---

### Experiment 20 — Guided exploration in danger

**Motivation**: prompted by reading a prior team's report for strategy only
(Ernst & Striebel 2021, "maverick" — professor's permission, no code
touched — see Experiment 17). They mixed a heuristic policy into their
epsilon-random fraction instead of pure uniform random and reported faster
convergence. Applied narrowly: uniform-random exploration while in danger
mostly means walking into the blast, which is exactly the training-data
skew Experiment 10/11 diagnosed (bad bombing outcomes dominating the
replay buffer).

**Change**: `QLEARN_GUIDED_EXPLORE` (default off). When set, an
epsilon-random step taken while `danger=1` follows the already-computed
committed escape direction with probability 0.7, uniform-random otherwise.
`danger=0` exploration is unchanged. Inert at eval (`self.train=False` means
`epsilon=0`, so `was_random` is never `True` regardless of this flag).

**Setup**: Experiment 18's exact protocol (`classic`, train=1000, eval=200),
`QLEARN_GUIDED_EXPLORE=1`, 5 seeds.

**Full 5-seed eval**:

| seed | bombs/round | crates/round | coins/round | suicide rate | avg steps |
|---|---|---|---|---|---|
| 0 | 1.17 | 4.38 | 0.20 | 20.0% | 324.5 |
| 1 | 1.01 | 3.83 | 0.14 | 11.0% | 358.2 |
| 2 | 3.52 | 14.60 | 0.92 | 3.0% | 388.6 |
| 3 | 3.33 | 11.53 | 0.55 | 44.0% | 231.3 |
| 4 | 1.86 | 5.51 | 0.19 | 7.0% | 376.1 |

**Aggregate**: bombs/round 2.18±1.18 (vs Exp 18 baseline: 1.95±0.52),
crates/round 7.97±4.81 (vs 7.09±2.01), coins/round 0.40±0.33 (vs
0.26±0.11), suicide rate 17.0%±16.0% (vs 19.0%±11.0%), avg steps
335.73±63.14 (vs 328.25±41.28).

`BOMB` row (seed 0): `bias=-0.440, safe_bomb=+0.751, crate_payoff=-0.638,
crate_dist=+0.139, payoff_x_safe=+1.807`.

**Read — corrected**: at n=5 with standard deviations this large, these
differences from Experiment 18 are **not statistically distinguishable** —
coins/round 0.40±0.33 vs 0.26±0.11 overlap heavily, and the entire apparent
"improvement" is carried by two outlier seeds (2 and 3, which also pull the
suicide-rate spread to its widest yet, 3-44%). This is **inconclusive, not
a mean improvement** — corrected from an earlier draft of this entry that
called it one. Re-run at 10 seeds as Experiment 22 to actually resolve
this.

**Observation (not a conclusion)**: `crate_payoff`'s weight reverted to
negative (-0.638) here, after Experiment 19 had resolved it to +0.171 at 5x
the training rounds. Plausibly guided exploration changed the training
distribution again in a way that reintroduced some of the original
confound (Experiment 11) — flagged for whoever revisits this, not
concluded here.

---

### Experiment 22 — Resolving Experiment 20 at 10 seeds

**Setup**: Experiment 18's exact protocol, 10 seeds instead of 5, both
arms. Baseline is also a direct re-run/replication check of Experiment 18.

**Baseline** (`QLEARN_GUIDED_EXPLORE` unset), 10 seeds: bombs/round
1.90±0.79, crates/round 6.93±2.92, coins/round 0.29±0.16, suicide rate
16.0%±11.0%, avg steps 340.57±43.57, invalid/round 0.06±0.06. Closely
replicates Experiment 18's 5-seed numbers (1.95±0.52 / 7.09±2.01 /
0.26±0.11 / 19.0%±11.0%) — consistent, not a fluke of the smaller sample.

**Guided** (`QLEARN_GUIDED_EXPLORE=1`), 10 seeds: bombs/round 2.18±0.88,
crates/round 8.25±3.66, coins/round 0.40±0.26, suicide rate 17.0%±12.0%,
avg steps 336.31±44.81, invalid/round 0.08±0.08.

**Read**: at n=10 the standard deviations are much closer between arms
than at n=5 (bombs std 0.88 vs 0.79, crates 3.66 vs 2.92, coins 0.26 vs
0.16 — compare to n=5's guided-arm stds of 1.18/4.81/0.33, roughly double
the baseline's). This resolves Experiment 20: guided exploration gives a
real, modest, consistent improvement in engagement (+38% coins/round,
+19% crates/round, +15% bombs/round) at essentially no cost in suicide
rate (17.0% vs 16.0%, well within noise). The n=5 result wasn't wrong, it
was just too small a sample to separate a real effect from two outlier
seeds — with 5 more seeds the outliers stopped dominating the mean.

---

## Next steps

1. ~~Investigate seed-to-seed suicide rate variance in Experiment 18~~ —
   resolved by Experiment 19: more training data doesn't narrow it, this
   looks like a genuine ceiling for the current feature set/reward
   structure.
2. Disaggregate the shaping ablation (INVALID_ACTION-only vs
   directional-only vs danger/bomb-safety-only) now that Task 2's reward
   table has several interacting shaping terms.
3. ~~Guided exploration~~ — implemented as Experiment 20; inconclusive at
   n=5 (see that entry). Re-run at 10 seeds as Experiment 22.
4. Re-validate whether prioritized replay is still needed now that the
   escape-commitment fix has changed the underlying dynamics substantially
   — the original diagnosis (Experiment 10/11) predates this fix.
5. ~~Second model~~ — done: SARSA (on-policy TD), Experiment 21. Directional
   support for the on-policy-conservatism hypothesis on Task 2, not a
   decisive win given the variance; matches Q-learning closely on Task 1
   as expected (no danger decisions to differ over there).
6. Now that both models exist: report-writing groundwork — the two-model
   comparison (Experiment 21) plus the systematic Task 1→2 progression
   across this whole log is close to what section 4-6 of the final report
   needs. Worth revisiting whether prioritized replay (item 4) or Task 3
   (opponents) is the better use of remaining time before the deadline.

---

## Agent: `sarsa_agent`

On-policy SARSA, the second model — `agent_code/sarsa_agent/`.
`callbacks.py` is byte-for-byte identical to `qlearn_agent`'s (same 30
features as of Experiment 24, same target/escape/opponent commitment
mechanisms). `train.py` keeps the same reward table and hyperparameters
(`alpha=0.01, gamma=0.9, epsilon: 0.3->0.05, replay batch=8, target sync
every 500 updates`). The
only algorithmic difference is the TD target: `Q_target(s', a'_actual)` —
the value of whatever action the policy actually takes next (epsilon-random
draws included) — instead of Q-learning's `max_a' Q_target(s', a')`.

**Deferred updates**: SARSA's target needs `a'`, which isn't known until
the *next* `act()` call chooses it — one step later than Q-learning can
update. Each `game_events_occurred`/`end_of_round` call first flushes
whatever transition is still pending, using its own `self_action`/
`last_action` parameter as `a'` (the actual action the framework recorded,
never recomputed or assumed), *then* stores its own transition as the new
pending one. The terminal step of a round updates directly with
`target=reward` (no bootstrap, no deferral). Replay entries are 6-tuples
`(s, a, r, s', a', terminal)`.

**Deferral verified correct before running anything** (per the task): ran
3 rounds with a debug line logging every flush's `(pending_action,
next_action)`, cross-referenced against the trace log's real action
sequence. Confirmed: the flush processing step *t*'s call always pairs
step *(t-1)*'s action with step *t*'s own action — never a premature or
recomputed one. Also confirmed at a round boundary (agent self-killed at
step 16): the trailing pending transition (`WAIT`→`BOMB`) was flushed
correctly with `next_action=BOMB` *before* that same `BOMB` transition was
finalized terminally (`target=reward=-6.0`, no bootstrap) as part of the
same `end_of_round` call, and the following round started with a clean
`self.pending=None` — no cross-round contamination.

---

### Experiment 21 — SARSA vs Q-learning (hypothesis, written before running)

**Hypothesis**: SARSA is on-policy, so its value estimates account for the
epsilon-random exploration the agent will actually perform during the
remaining trajectory, rather than assuming optimal (greedy) continuation
the way Q-learning's `max` does. Near a ticking bomb, where a single wrong
step during the ~4-step escape window is fatal, this should make SARSA
*more conservative* about bombing than Q-learning: it should predict a
**lower suicide rate**, plausibly accompanied by **fewer bombs dropped and
fewer crates destroyed** (a more cautious policy engages less, not just
survives more per engagement). This is the textbook cliff-walking-style
prediction for on-policy vs off-policy TD control, applied to this task's
own cliff (a bomb's blast).

**Setup**:
- Task 2: `classic`, train=1000, eval=200, 10 seeds
- Task 1: `coin-heaven`, train=300, eval=100, 10 seeds
- Q-learning comparison: Experiment 22 (10-seed re-run of Experiment 18) for
  Task 2; a matching 10-seed `qlearn_agent` run for Task 1
- Board and agent seeds controlled throughout, same harness
  (`scripts/run_experiment.py`)

**Results — Task 2** (`classic`, 10 seeds):

| metric | SARSA | Q-learning (Exp 22) |
|---|---|---|
| bombs/round | 1.59 ± 1.31 | 1.90 ± 0.79 |
| crates/round | 6.37 ± 5.15 | 6.93 ± 2.92 |
| coins/round | 0.35 ± 0.30 | 0.29 ± 0.16 |
| suicide rate | 12.0% ± 13.0% | 16.0% ± 11.0% |
| avg steps | 355.56 ± 47.80 | 340.57 ± 43.57 |

**Results — Task 1** (`coin-heaven`, 10 seeds):

| metric | SARSA | Q-learning |
|---|---|---|
| completion rate | 0.80 ± 0.42 | 0.59 ± 0.48 |
| avg coins/50 | 41.06 ± 18.85 | 33.39 ± 20.40 |
| steps\|completed | 125.58 ± 0.83 | 125.59 ± 0.43 |

**`BOMB` weight row, seed 0** — SARSA: `bias=-0.909, safe_bomb=+1.474,
crate_payoff=-0.332, crate_dist=-0.541, payoff_x_safe=+1.552`. Q-learning
(Experiment 18/19, seed 0, for comparison): `bias≈-0.76 to -0.84,
safe_bomb≈+1.23 to +1.24, payoff_x_safe≈+1.08 to +1.49`. SARSA's
`safe_bomb` weight (+1.474) is the strongest seen in any experiment with
this feature set — consistent with the hypothesis: an on-policy target
that accounts for the actual (noisy) continuation should make the safety
feature more decision-relevant than an off-policy target that assumes
optimal escape.

**Read**: **directionally confirms the hypothesis, but not decisively.**
SARSA does show lower bombs/round (-16%), lower suicide rate (-4
percentage points), and longer survival (+4%) on Task 2, matching
"more conservative near bombs." But SARSA's variance is far higher than
Q-learning's on every Task 2 metric (bombs std 1.31 vs 0.79, crates 5.15
vs 2.92, suicide 13% vs 11%) — driven by a real seed split, not noise
inflation: some seeds barely engage (0.12-0.14 bombs/round) while others
engage heavily (3.52 bombs/round, comparable to Q-learning's upper range).
Given the overlapping distributions, this is a real directional signal,
not a decisive win — the means point the predicted way but I would not
claim statistical separation without a formal test.

On Task 1, where there's no bombing/danger decision to differ over, SARSA
and Q-learning are close on both algorithms' own terms (both around
125.5 steps to complete), but SARSA's completion rate (0.80) came out
clearly higher than Q-learning's freshly-run 10-seed number (0.59) — note
this is *lower* than Q-learning's own earlier 5-seed Experiment 8 result
(0.80±0.45), meaning Q-learning's Task 1 seed-variance is real and the
5-seed sample undersold it; SARSA's 10-seed number is more directly
comparable. This Task 1 gap isn't predicted by the on-policy/off-policy
hypothesis (which is specifically about danger/bombing) and is more likely
just favorable seed variance for SARSA here, not a general Task 1
advantage — flagged as an observation, not explained by the hypothesis
this experiment was designed to test.


---

### Experiment 23 — Evaluation vs. provided agents (not an attempt at Tasks 3/4)

**Upfront caveat, stated before any results**: neither `qlearn_agent` nor
`sarsa_agent` has any opponent-aware features. `state_to_features` never
reads `game_state['others']` for anything except treating opponent
positions as static occupied tiles in the BFS pathfinding/danger checks —
there is no feature that encodes "where is an opponent", "is an opponent
near", or anything resembling opponent-directed behavior. Both agents were
trained exclusively on Task 2 (`classic`, single agent, no opponents). This
experiment measures **where a Task-2-trained, opponent-blind agent stands
the moment opponents appear** — it is not an attempt at Tasks 3 or 4, and
poor results are expected and reported as such, not softened.

**Setup**: evaluation only, no training, no new features. Existing
checkpoints only — `qlearn_agent`'s reproduces Experiment 18's exact
config (seed 0, 1000 rounds, `classic`, default hyperparameters); `sarsa`'s
reproduces Experiment 21's. Both retrained fresh from the same seeds to
restore the checkpoint state (the actual weight files had since been
overwritten by later experiments in this session; same code, same seed,
same protocol — deterministic reproduction, not a new/different run).
`classic`, 200 rounds, 5 seeds (board seeds disjoint from every prior range
used this session), `self.train=False` for every agent (no `--train` flag).
Three opponent configurations per agent:
- Task 3 setup: `+ peaceful_agent + coin_collector_agent`
- Task 4 head-to-head: `+ rule_based_agent`
- Tournament configuration: `+ 3x rule_based_agent`

**Methodology note**: `--save-stats` only gives *cumulative* per-agent
score over the whole 200-round run, not per-round, so win/draw/loss is
reconstructed from `game.log` (the only two score-affecting events in the
framework, verified against `environment.py`, are coin pickup and
opponent-kill — both logged with parseable messages) and cross-checked
against the JSON's cumulative score every single run (`score_check_ok`
field) to catch any parsing gap. Similarly, `GOT_KILLED` fires on both
self-kills and opponent-kills with no distinguishing stat in the saved
JSON, so "deaths by opponent" vs. "self-kill" is split by grepping
`game.log`'s two distinct log messages (`"blown up by own bomb"` vs.
`"blown up by agent <X>'s bomb"`) for our agent's name specifically.

**Results.** All 30 runs (2 agents x 3 configs x 5 seeds) passed the
score-reconstruction cross-check (`score_check_ok=True` every time) — the
win/draw/loss and score numbers below are verified consistent with the
saved JSON, not just log-parsed in isolation.

**Aggregate, `qlearn_agent`** (mean ± std over 5 seeds, 1000 rounds/config):

| config | our score/round | W/D/L (of 1000) | win rate | self-kills | deaths by opp. | opp. killed | bombs/round | crates/round | coins/round | suicide rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Task 3 (peaceful+coin_collector) | 0.791±0.146 | 20/13/967 | 2.0% | 64.0±6.5 | 34.8±8.5 | 9.2±3.9 | 3.97±0.22 | 13.29±0.77 | 0.561±0.077 | 32.0% |
| Task 4 (1x rule_based) | 0.617±0.080 | 18/11/971 | 1.8% | 84.4±6.7 | 50.4±7.8 | 3.0±0.7 | 4.08±0.28 | 13.43±0.93 | 0.542±0.077 | 42.2% |
| Tournament (3x rule_based) | 0.568±0.101 | 12/3/985 | 1.2% | 75.4±4.8 | 101.2±4.9 | 5.6±3.0 | 3.68±0.15 | 11.55±0.38 | 0.428±0.042 | 37.7% |

**Aggregate, `sarsa_agent`** (mean ± std over 5 seeds, 1000 rounds/config):

| config | our score/round | W/D/L (of 1000) | win rate | self-kills | deaths by opp. | opp. killed | bombs/round | crates/round | coins/round | suicide rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Task 3 (peaceful+coin_collector) | 1.228±0.127 | 38/12/950 | 3.8% | 98.8±3.3 | 9.8±3.0 | 6.8±1.6 | 6.27±0.13 | 18.56±0.26 | 1.058±0.087 | 49.4% |
| Task 4 (1x rule_based) | 1.345±0.134 | 52/13/935 | 5.2% | 116.4±4.7 | 26.0±7.0 | 2.4±1.8 | 6.94±0.07 | 20.92±0.37 | 1.285±0.116 | 58.2% |
| Tournament (3x rule_based) | 1.084±0.150 | 33/19/948 | 3.3% | 121.6±6.9 | 49.2±3.5 | 7.2±2.8 | 5.68±0.09 | 15.24±0.70 | 0.904±0.097 | 60.8% |

Per-seed breakdowns (score, W/D/L, self-kills/deaths-by-opponent per 200
rounds) for all 6 agent x config combinations are in the raw run output —
omitted here for length, but every aggregate above is a straight mean over
exactly those 5 numbers, and none of them hide a single outlier seed
dominating the pattern (checked directly: e.g. qlearn tournament
deaths-by-opponent per seed was 103/97/109/99/98 — tight, not one wild
seed).

**The key split, as designed to answer**: for `qlearn_agent`, self-kills
dominate deaths-by-opponent in Task 3 (64 vs 35) and Task 4 (84 vs 50), but
**this flips in the tournament config** — deaths-by-opponent (101.2) clearly
exceed self-kills (75.4) for the first time. Against 3 simultaneous
aggressive opponents, "blindness to other agents" overtakes the known
escape weakness as the dominant failure mode. For `sarsa_agent`, self-kills
stay dominant in *every* config, including the tournament (121.6 vs 49.2) —
its failure mode is overwhelmingly "kills itself" regardless of how many
opponents are present.

**A genuine reversal worth flagging plainly**: Experiment 21 found SARSA
*more conservative* than Q-learning in the single-agent setting (lower
suicide rate, fewer bombs). Here, with opponents present, **SARSA's
suicide rate is 10-19 percentage points *higher* than Q-learning's in every
single config** (49.4% vs 32.0%, 58.2% vs 42.2%, 60.8% vs 37.7%), and
SARSA drops roughly 1.5-1.7x more bombs/round than Q-learning throughout.
The single-agent-trained conservatism advantage does not transfer to the
presence of opponents — plausibly because neither agent's training ever
included another agent's bombs, blocked tiles, or crate destruction, so
SARSA's on-policy value estimates (tuned to *its own* exploration noise,
never to opponent-caused disruption) may be poorly calibrated for a kind of
unpredictability it never trained against. This is offered as the most
plausible mechanism, not a verified one — no experiment here isolates it.
Despite the higher suicide rate, SARSA's much higher engagement (more
bombs/crates/coins per round) nets it a consistently *better* score/round
and win rate than Q-learning in all three configs (e.g. tournament: 1.084
vs 0.568 score/round, 3.3% vs 1.2% win rate) — it dies more often but
scores enough before dying to come out ahead on the tournament's actual
criterion (total score).

**Win rates are low but non-zero** (1.2-5.2%) for both agents in every
config — consistent with "opponent-blind, Task-2-trained agents facing
scripted/skilled opponents," as expected going in, not a surprise.

---

**Observation, not from this experiment specifically**: the seed-to-seed
engagement/suicide-rate split (some seeds barely engage, others engage
heavily) has now shown up in Q-learning (Experiments 19, 22) *and* SARSA
(Experiment 21) under the same Task 2 single-agent protocol — the same
qualitative pattern across two different learning algorithms sharing only
the feature set and reward structure. This suggests the split is more
likely a property of the **environment or feature set** (e.g. some board
layouts/starting positions are structurally more forgiving for this
feature set's escape logic than others) than of either learning algorithm
specifically — worth keeping in mind before attributing it to anything
algorithm-specific in the report.


---

### Experiment 24 — Opponent-aware features + training on opponents

**Motivation**: Experiment 23 showed deaths-by-opponent (101.2) overtaking
self-kills (75.4) for `qlearn_agent` specifically in the tournament config —
opponent-blindness becomes the dominant failure mode exactly in the setting
that counts. Both agents were also, at that point, still trained
single-agent only. This experiment adds opponent-aware features (see the
feature list above) and an opponent-weighted `KILLED_OPPONENT` reward, then
retrains both agents against opponents for the first time, to test whether
either change — or both together — shifts that ratio back.

**Setup**: two new checkpoints per agent, both `classic`, 1000 rounds,
seed 0, opponent-aware (30-feature) code throughout, pre-existing Task-2
checkpoints (`model_task2_baseline.pt`) preserved untouched:
- `model_vs_peaceful_coin.pt` — trained against `peaceful_agent +
  coin_collector_agent` (`--train 1`)
- `model_vs_rule_based.pt` — trained against `rule_based_agent` (`--train 1`)

Evaluated under Experiment 23's exact protocol for direct comparability:
`classic`, 200 rounds, 5 seeds, `self.train=False`, the same three opponent
configs (Task 3 setup, Task 4 head-to-head, tournament), and the same
log-parsing methodology (win/draw/loss and self-kill/deaths-by-opponent
reconstructed from `game.log`, cross-checked against `--save-stats`'s
cumulative score every run). This run is an independent re-implementation
of that methodology (`scripts/eval_vs_opponents.py`), not a reuse of an
undisclosed prior script, matching Experiment 23's described approach.

**A bug caught and fixed before trusting any result**: `main.py`'s default
`--log-dir` is a single fixed path, and `environment.py` opens `game.log`
there with `mode="w"` per subprocess. Running qlearn's and sarsa's eval jobs
in parallel (to save wall-clock time) had both writing to that same file
concurrently, corrupting it continuously — caught because `score_check_ok`
came back `False` on every single one of the first 30 runs, not because of
any visible crash. Fixed by giving each `evaluate_one()` call its own
`--log-dir`, keyed by run tag; verified with a 5-round smoke test
(`score_check_ok=True`) before discarding the corrupted results and
re-running the full evaluation. All results below passed the cross-check
(`score_check_ok=True`) on every one of the 60 runs (2 agents × 2
checkpoints × 3 configs × 5 seeds).

**Aggregate, `qlearn_agent`** (mean ± std over 5 seeds, 200 rounds/config;
Task-2 baseline row is Experiment 23's, reproduced for reference):

| config | training | score/round | win rate | self-kills | deaths by opp. | opp. killed | bombs/round | crates/round | coins/round |
|---|---|---|---|---|---|---|---|---|---|
| Task 3 | Task-2 only (baseline) | 0.791±0.146 | 2.0% | 64.0±6.5 | 34.8±8.5 | 9.2±3.9 | 3.97±0.22 | 13.29±0.77 | 0.561±0.077 |
| Task 3 | vs peaceful+coin_collector | 0.778±0.119 | 2.0% | 25.0±1.6 | 52.8±7.7 | 15.0±4.6 | 2.77±0.10 | 8.49±0.34 | 0.403±0.030 |
| Task 3 | vs rule_based | 1.552±0.229 | 6.0% | 135.2±4.0 | 30.2±4.4 | 10.0±2.7 | 5.17±0.51 | 19.07±1.99 | 1.302±0.179 |
| Task 4 | Task-2 only (baseline) | 0.617±0.080 | 1.8% | 84.4±6.7 | 50.4±7.8 | 3.0±0.7 | 4.08±0.28 | 13.43±0.93 | 0.542±0.077 |
| Task 4 | vs peaceful+coin_collector | 0.684±0.061 | 3.7% | 40.0±4.4 | 77.4±5.3 | 7.8±1.5 | 3.24±0.13 | 9.83±0.50 | 0.489±0.066 |
| Task 4 | vs rule_based | 1.312±0.210 | 7.0% | 138.0±5.8 | 46.2±5.3 | 4.0±2.4 | 4.99±0.40 | 18.24±1.49 | 1.212±0.151 |
| Tournament | Task-2 only (baseline) | 0.568±0.101 | 1.2%* | 75.4±4.8 | 101.2±4.9 | 5.6±3.0 | 3.68±0.15 | 11.55±0.38 | 0.428±0.042 |
| Tournament | vs peaceful+coin_collector | 0.571±0.085 | 0.6% | 42.6±8.2 | 137.2±9.7 | 7.2±2.0 | 2.76±0.14 | 7.77±0.39 | 0.391±0.052 |
| Tournament | vs rule_based | 0.917±0.110 | 2.4% | 130.0±5.8 | 66.2±4.4 | 5.8±3.6 | 3.67±0.22 | 12.46±0.87 | 0.772±0.058 |

**Aggregate, `sarsa_agent`** (same format):

| config | training | score/round | win rate | self-kills | deaths by opp. | opp. killed | bombs/round | crates/round | coins/round |
|---|---|---|---|---|---|---|---|---|---|
| Task 3 | Task-2 only (baseline) | 1.228±0.127 | 3.8% | 98.8±3.3 | 9.8±3.0 | 6.8±1.6 | 6.27±0.13 | 18.56±0.26 | 1.058±0.087 |
| Task 3 | vs peaceful+coin_collector | 0.827±0.056 | 2.3% | 80.6±8.0 | 19.4±8.3 | 4.4±1.5 | 3.79±0.26 | 13.84±0.92 | 0.717±0.056 |
| Task 3 | vs rule_based | 1.443±0.130 | 4.6% | 93.6±5.9 | 40.4±4.9 | 9.6±3.0 | 5.66±0.11 | 19.92±0.63 | 1.203±0.068 |
| Task 4 | Task-2 only (baseline) | 1.345±0.134 | 5.2% | 116.4±4.7 | 26.0±7.0 | 2.4±1.8 | 6.94±0.07 | 20.92±0.37 | 1.285±0.116 |
| Task 4 | vs peaceful+coin_collector | 0.733±0.087 | 1.7% | 92.0±9.8 | 63.6±4.2 | 1.2±1.3 | 3.98±0.23 | 14.41±0.86 | 0.703±0.108 |
| Task 4 | vs rule_based | 1.169±0.188 | 2.8% | 95.8±7.2 | 88.2±6.9 | 1.2±1.3 | 5.24±0.18 | 18.41±0.87 | 1.139±0.163 |
| Tournament | Task-2 only (baseline) | 1.084±0.150 | 3.3%* | 121.6±6.9 | 49.2±3.5 | 7.2±2.8 | 5.68±0.09 | 15.24±0.70 | 0.904±0.097 |
| Tournament | vs peaceful+coin_collector | 0.601±0.075 | 0.8% | 78.4±4.3 | 93.8±3.3 | 2.0±2.0 | 3.56±0.11 | 11.76±0.46 | 0.551±0.050 |
| Tournament | vs rule_based | 0.952±0.117 | 0.5% | 74.6±4.4 | 123.2±2.8 | 4.6±2.2 | 4.34±0.21 | 13.98±0.59 | 0.837±0.074 |

**The self-kill vs. deaths-by-opponent ratio, tournament config specifically
(the question this experiment was designed to answer)**:

| agent | training | self-kills | deaths by opp. | ratio (self/opp) | dominant failure |
|---|---|---|---|---|---|
| qlearn | baseline | 75.4 | 101.2 | 0.75 | opponent |
| qlearn | vs peaceful+coin_collector | 42.6 | 137.2 | 0.31 | opponent (worse) |
| qlearn | vs rule_based | 130.0 | 66.2 | **1.96** | **self (flipped back)** |
| sarsa | baseline | 121.6 | 49.2 | 2.47 | self |
| sarsa | vs peaceful+coin_collector | 78.4 | 93.8 | 0.84 | opponent (flipped) |
| sarsa | vs rule_based | 74.6 | 123.2 | 0.61 | opponent (worse) |

**Read — the ratio shifts, but the two agents shift in opposite
directions, and neither shift is "features alone did it."**

For `qlearn_agent`, adding the opponent-aware *features* without training
against an actual opponent (`vs peaceful+coin_collector`) does not fix
opponent-blindness — it makes it worse (deaths-by-opponent 137.2, up from
101.2). What restores self-kill dominance is training against a real,
bomb-dropping adversary (`vs rule_based`): the ratio flips from 0.75 back
to 1.96, matching the Task 3/4 baseline pattern where self-kills always
led. But this "fix" is not free — self-kills roughly doubled in *every*
config versus baseline (Task 3: 64→135, Task 4: 84→138, tournament:
75→130), and specifically in tournament, self-kills-per-bomb-dropped also
rose (baseline ≈75.4 self-kills / 736 bombs ≈10.2%; vs-rule_based ≈130.0 /
734 bombs ≈17.7% — nearly the same bomb volume, a higher self-kill rate per
bomb). Plausible mechanism, not verified here: training against an actual
bomber means the agent now regularly faces a *second* bomb appearing after
it has already committed to an escape route it computed as safe against
only the bombs known at that moment — the `_can_escape_own_bomb` safety
search sees currently-placed bombs, not bombs the opponent has not dropped
yet. `KILLED_OPPONENT`'s reward may also be pulling the agent into more
contested, bomb-dense positions (bombs/round rose from 2.77 to 3.67 in
tournament versus the `vs peaceful+coin_collector` checkpoint), which
mechanically raises exposure to exactly this kind of stacked-danger
situation.

For `sarsa_agent`, the picture is the *reverse*. It starts self-kill
dominant at baseline (2.47) and *both* new checkpoints — with or without
opponents in training — push it further toward opponent-dominance (0.84,
then 0.61), the opposite direction from qlearn under identical features and
rewards. SARSA's self-kills actually *fell* in every config relative to
baseline (Task 3: 98.8→80.6→93.6, Task 4: 116.4→92.0→95.8, tournament:
121.6→78.4→74.6), while deaths-by-opponent rose sharply in every config
(most starkly tournament: 49.2→93.8→123.2). Training against rule_based_agent
did not reproduce qlearn's self-kill rebound. A plausible reading, offered
as speculative and not verified by any experiment here: SARSA's on-policy
target already factors in the agent's own future exploration noise, so the
opponent-awareness features may be read more conservatively (avoid the
opponent rather than engage), while qlearn's `max`-based target, paired
with the new `KILLED_OPPONENT` reward, may be optimistically pursuing
opponent-adjacent bombing plays its own escape logic can't reliably survive
once a second bomber is on the board.

**A second bug, caught by a comment on this doc, not by any crash or by
`score_check_ok`**: the tournament win/draw/loss numbers originally
reported here were wrong — inflated to 34-54% win rate, when the corrected
number is 0.5-2.4%, in line with Task 3/4 and with Experiment 23's
baseline. `score_check_ok` only cross-checks *our own* score against the
JSON — it says nothing about whether opponent scores, or the win/draw/loss
classification built from them, are correct, and this went unnoticed
through the whole write-up above until it was pointed out directly.

Root cause: `environment.py`'s `setup_agents` (`environment.py:341-346`)
disambiguates duplicate agent-dir names by suffixing them in play order —
three `rule_based_agent` entries in the tournament config become
`rule_based_agent_0/_1/_2` in `game.log` and `--save-stats`, confirmed
directly in a real log line (`Agent <rule_based_agent_0> chose action
RIGHT...`). `eval_vs_opponents.py`'s `CONFIGS["tournament"]` is the raw,
*undisambiguated* list (`["rule_based_agent"] * 3`), and `evaluate_one()`
built `all_names` straight from it. `win_draw_loss()`'s
`scores = {name: r.get(name, 0) for name in all_names}` then collapsed the
three duplicate keys into one dict entry that never matched any real
per-round key — so "the opponent's score" silently read as `0` every
round, for every seed, in every tournament-config run this experiment
produced. Consequence: a loss became structurally impossible (`L=0` on
every single tournament round-set — the one number that should have
prompted a second look and didn't), a round was scored a draw whenever
*our* score was `0` regardless of what the real leader scored, and a win
was recorded whenever our score was `>0` at all, regardless of whether an
opponent outscored us. Verified concretely on `qlearn_agent`'s
`vs_rule_based` tournament seed-0 log: round 1 (us 0, opponents 3/4/2 — a
real loss) was classified `D`; round 3 (us 3, opponents 7/3/1 — also a real
loss) was classified `W`.

Fixed by adding `disambiguate_names()` (reproduces `setup_agents`'
suffixing exactly) and using its output as `all_names` in `evaluate_one()`.
`parse_game_log()` was never affected — it reads names straight off
`game.log`, never off the `all_names` parameter it's passed (which,
on inspection, it doesn't actually use). Because the raw `game.log` and
`--save-stats` files from every run were still on disk, no games needed to
be rerun: `scripts/reprocess_exp24.py` recomputes win/draw/loss from the
existing data with the fix applied. Task 3/4 numbers came back
byte-identical to the original run (no duplicate opponent names there, so
nothing to fix) — the corrected tournament win rates above are the only
numbers this bug changed; `self_kills`/`deaths_by_opponent`/
`opponents_killed`/`score_per_round` were never affected by it (all keyed
directly on `our_name`, never on the broken merged dict), so the ratio
table and the read above it stand as written.

**Experiment 23's baseline tournament win rates (marked `*` above) could
not be re-verified** — they were produced by a different, prior script
whose source is not available to inspect, and its `game.log` was not
preserved per-run (each run overwrote a single shared log file, before
this experiment's per-tag `--log-dir` fix existed), so there is nothing
left to reprocess. Given this bug's root cause is generic to any tournament
win/draw/loss reconstruction against duplicate-named opponents, those two
numbers should be treated as unverified, not trusted at face value, until
someone re-derives them from a fresh run.

**Score and win rate, for context (not the question this experiment was
designed to answer, but relevant to whether either change is worth
keeping)**, corrected: tournament win rate does *not* meaningfully move for
either agent or either new checkpoint (qlearn 0.6%/2.4%, sarsa 0.8%/0.5%,
against an unverified 1.2%/3.3% baseline) — consistent with Task 3/4's own
low, largely-flat win rates, and with "win" against three simultaneously
aggressive `rule_based_agent`s remaining a high bar throughout. Nothing
here supports reading either change as a tournament-win-rate improvement;
the self-kill/deaths-by-opponent ratio (unaffected by this bug) remains the
only clear signal from this experiment.

**Net read on the two changes** requested at the top of this experiment —
opponent-aware features, and training on opponents — is that they are not
separable from these results and do not simply "fix" opponent-blindness:
features without opponent training made qlearn's opponent-blindness worse;
opponent training fixed qlearn's ratio but at the cost of doubling its
self-kill rate; and the same combination pushed sarsa in the opposite
direction entirely. Whatever is driving qlearn's self-kill spike under
opponent training (most plausibly, escape safety no longer accounting for
a second bomber) is the natural next thing to isolate before either
checkpoint is called an improvement over Task-2-only baseline.

---

### Experiment 25 — Stacked-bomb hypothesis test, corrected baseline, submission decision table

**Setup notes common to all three parts**:

- **Zero-padded baseline checkpoints**: `model_task2_baseline.pt` is a
  `(6, 23)` weight array; current code computes 30 features. Padding it to
  `(6, 30)` with zero columns for indices 23-29 (the Experiment 24
  opponent features) is provably behavior-equivalent to the true 23-feature
  policy — `Q(s,a) = w[a]·features(s)`, and a zero weight on the new
  indices contributes exactly 0 regardless of their value, as long as
  Experiment 24 only *appended* features 23-29 without altering the
  computation of 0-22 (confirmed by re-reading `state_to_features` — the
  opponent block is a pure, non-mutating addition after all prior feature
  writes). Verified numerically (`padded @ f == w @ f[:23]` for random
  30-dim `f`), not just argued. `model_task2_baseline_padded.pt` created
  for both agents; regenerated baseline numbers below reproduce Experiment
  23's original (self-kills 75.4→73.2, deaths-by-opp 101.2→102.2 for
  qlearn; 121.6→120.4 / 49.2→50.0 for sarsa — within seed noise), which is
  itself a second, independent confirmation the padding is faithful.
- **Self-kill bucket classification** (`scripts/classify_selfkills.py`):
  matches each of our self-kills to the specific bomb that caused it via
  `game.log`'s drop/explode/self-kill lines (`environment.py`'s exact
  formats), relying on the game's `bombs_left` mechanic guaranteeing at
  most one live bomb per agent at a time — so the most recent explosion
  logged for an owner is unambiguously that owner's most recent drop, no
  position-matching heuristics needed. Bucket (a): at least one opponent
  bomb-drop logged in `(t_drop, t_death]`. Bucket (b): none. 0 unmatched
  self-kills across all 20 logs processed (1332 total self-kills
  classified) — every case resolved. Cross-checked against each run's own
  reported `self_kills` aggregate (bucket totals sum to it exactly in all
  4 conditions) before trusting the split.
- **Corrected tournament win/draw/loss**: `scripts/eval_baseline_tournament.py`
  reuses `evaluate_one()` from `scripts/eval_vs_opponents.py`, which now
  includes the `disambiguate_names()` fix from earlier this experiment.

---

#### Part 1 — raw numbers: self-kill bucket split

| condition | bucket (a): opp. bomb during our window | bucket (b): no opp. bomb | n (=self-kills) | (a) share |
|---|---|---|---|---|
| qlearn, baseline | 186 | 180 | 366 | 50.8% |
| qlearn, vs_rule_based | 339 | 311 | 650 | 52.2% |
| sarsa, baseline | 399 | 203 | 602 | 66.3% |
| sarsa, vs_rule_based | 263 | 110 | 373 | 70.5% |

Per-seed breakdown (a, b, unmatched), 5 seeds each:
- qlearn baseline: `[(40,38,0), (39,33,0), (39,34,0), (37,43,0), (31,32,0)]`
- qlearn vs_rule_based: `[(68,64,0), (72,65,0), (70,63,0), (59,65,0), (70,54,0)]`
- sarsa baseline: `[(92,40,0), (67,49,0), (85,46,0), (83,29,0), (72,39,0)]`
- sarsa vs_rule_based: `[(47,24,0), (50,22,0), (49,26,0), (64,18,0), (53,20,0)]`

Per-seed mean counts and the baseline→vs_rule_based change:

| agent | bucket | baseline (mean/seed) | vs_rule_based (mean/seed) | change | % change |
|---|---|---|---|---|---|
| qlearn | (a) | 37.2 | 67.8 | +30.6 | +82.3% |
| qlearn | (b) | 36.0 | 62.2 | +26.2 | +72.8% |
| sarsa | (a) | 79.8 | 52.6 | -27.2 | -34.1% |
| sarsa | (b) | 40.6 | 22.0 | -18.6 | -45.8% |

**Part 1 — interpretation, kept separate from the numbers above.**

The bucket split does not support the stacked-bomb hypothesis as the
mechanism behind qlearn's self-kill-rate increase, on two independent
readings of "dominates":

1. **Share, in isolation**: bucket (a) is a bare majority for qlearn under
   `vs_rule_based` (52.2%) — not a dominant failure mode by any reasonable
   reading, and barely different from qlearn's *own baseline* share
   (50.8%). Sarsa's share is more lopsided (70.5%), but its baseline share
   is already 66.3% — nearly as high *without* ever training against an
   opponent, so a high bucket-(a) share looks like a background property
   of the tournament environment (3 simultaneous bombers create frequent
   bomb-timing overlaps regardless of who is playing), not something
   `vs_rule_based` training specifically introduced.
2. **Where the actual increase came from (the more direct test)**: for
   qlearn, the agent whose self-kill rate increase motivated this
   experiment, both buckets grew by comparable percentages (+82.3% for (a),
   +72.8% for (b)) — bucket (a) accounts for only 54% of the net increase,
   barely more than an even split. If the stacked-bomb mechanism were the
   driver, the increase should have been concentrated in (a); it isn't.
   For sarsa, self-kills *fell* under `vs_rule_based` training (already
   established in Experiment 24), and that fall is similarly spread across
   both buckets, not concentrated in either.

**Conclusion: bucket (a) does not dominate, in the causally relevant sense,
for either agent — Part 2 (the escape-safety code changes and retraining)
is not warranted by this evidence and is skipped.** Whatever is actually
driving qlearn's self-kill-rate increase under opponent training remains
unexplained by the specific "second bomb invalidates a committed escape
route" mechanism flagged in Experiment 24 — that mechanism is real and
present (roughly half of all self-kills, in both conditions, do involve a
same-window opponent bomb), but it isn't what changed. A more likely
candidate, not tested here: `vs_rule_based` training's higher overall
engagement (more bombs, more crates, more coins per round across the board
— see Experiment 24's tables) may simply mean more total bomb-drop
decisions made under marginal safety margins, raising the self-kill count
roughly proportionally across every sub-cause rather than through one
specific new failure mode.

---

#### Part 3 — raw numbers: corrected baseline + submission decision table

**Regenerated tournament baseline** (padded `model_task2_baseline.pt`,
corrected win/draw/loss, `classic`, 200 rounds, 5 seeds — replaces
Experiment 24's `*`-flagged, unverified Experiment 23 baseline numbers):

| agent | score/round | win rate | self-kills | deaths by opp. | opp. killed | bombs/round | crates/round | coins/round |
|---|---|---|---|---|---|---|---|---|
| qlearn baseline (regenerated) | 0.583±0.018 | 1.3% | 73.2±6.6 | 102.2±7.3 | 4.6±1.1 | 3.674±0.090 | 11.346±0.378 | 0.468±0.037 |
| sarsa baseline (regenerated) | 1.120±0.133 | 4.0% | 120.4±10.3 | 50.0±8.5 | 7.0±4.7 | 5.693±0.267 | 15.264±0.892 | 0.945±0.066 |

All 10 runs (2 agents × 5 seeds) passed `score_check_ok=True`. Both numbers
land close to Experiment 23's original, now-corroborated figures (self-kills
75.4→73.2 and 121.6→120.4; deaths-by-opp 101.2→102.2 and 49.2→50.0; win
rate 1.2%→1.3% and 3.3%→4.0%) — the small differences are within seed
noise, not evidence of a problem with either the original numbers or the
padding method.

**The submission decision table** — all 4 candidate checkpoints, tournament
config (3× `rule_based_agent`), 200 rounds, 5 seeds, ranked by score/round:

| rank | agent | training | score/round | win rate | self-kills | deaths by opp. | opp. killed | bombs/round |
|---|---|---|---|---|---|---|---|---|
| 1 | sarsa | Task-2 only (baseline) | **1.120±0.133** | **4.0%** | 120.4±10.3 | 50.0±8.5 | 7.0±4.7 | 5.693±0.267 |
| 2 | sarsa | vs rule_based | 0.952±0.117 | 0.5% | 74.6±4.4 | 123.2±2.8 | 4.6±2.2 | 4.34±0.21 |
| 3 | qlearn | vs rule_based | 0.917±0.110 | 2.4% | 130.0±5.8 | 66.2±4.4 | 5.8±3.6 | 3.67±0.22 |
| 4 | qlearn | Task-2 only (baseline) | 0.583±0.018 | 1.3% | 73.2±6.6 | 102.2±7.3 | 4.6±1.1 | 3.674±0.090 |

**Part 3 — interpretation, kept separate from the numbers above.**

The plain, Task-2-only `sarsa_agent` checkpoint — never trained against an
opponent, no opponent-aware features contributing (weights on those
indices are exactly 0 by construction) — has both the highest score/round
*and* the highest win rate of all 4 candidates in the tournament config,
the setting that decides submission. Every change made in Experiments 24
and this one (opponent-aware features, opponent training, the
`KILLED_OPPONENT` reward) failed to produce a checkpoint that beats it on
either metric. This ranking is unaffected by either bug found this session
— it uses only the corrected win/draw/loss logic and numbers cross-checked
against `--save-stats` throughout.

This does not mean the opponent-aware work was wasted — `qlearn vs
rule_based` is a clear improvement over `qlearn baseline` on both metrics
(0.917 vs 0.583 score/round, 2.4% vs 1.3% win rate), and Experiment 24's
self-kill/deaths-by-opponent ratio work stands on its own. But on the
number that actually decides which agent gets submitted, the answer coming
out of this experiment is: **plain baseline SARSA, not either opponent-aware
variant** — worth stating plainly rather than assuming the more heavily
engineered checkpoint must be the better one.
