import json
import os
import pickle
from collections import deque
from typing import List

import numpy as np

import settings as s
import events as e
from .callbacks import (
    ACTIONS, MOVE_ACTIONS, MODEL_FILE, state_to_features, potential,
    get_blast_coords,
    IDX_TARGET_BASE, IDX_SAFE_BOMB, IDX_CRATE_PAYOFF, IDX_OPPONENT_DIR_BASE,
)

# Diagnostic instrumentation (see docs/experiments.md) -- off by default,
# does not affect normal training/eval. When on, logs every BOMB transition
# (was it epsilon-random or greedy, safe_bomb/crate_payoff feature values,
# reward) to a JSON file for offline analysis.
INSTRUMENT = os.environ.get("QLEARN_INSTRUMENT", "0") == "1"
INSTRUMENT_FILE = os.path.join(os.path.dirname(__file__), 'instrument.json')

# Ground-truth check for _can_escape_own_bomb (Experiment 13): every actual
# bomb drop (round, step, safe_bomb value at drop time) and every self-kill
# (round, step), so the false-positive rate -- safe_bomb=1 at drop time
# followed by a self-kill within BOMB_TIMER+2 steps -- can be computed
# offline instead of inferred from weights.
ESCAPE_CHECK_FILE = os.path.join(os.path.dirname(__file__), 'escape_check.json')

# Hyperparameters
# ALPHA is deliberately small: with REPLAY_BATCH_SIZE=8 there are 9 gradient
# updates per environment step (1 online + 8 replay). At the original
# ALPHA=0.1 this diverged -- a clean weight dump after "converged" training
# showed every action with a large positive bias (4-8) and non-diagonal,
# all-positive target-direction weights instead of the expected diagonal
# structure (see docs/experiments.md, Experiment 8). ALPHA=0.01 with a
# tighter, update-count-based target sync (below) fixed it.
ALPHA = 0.01           # learning rate
# Overridable for the gamma-sweep experiment (docs/experiments.md, Experiment
# 17) -- default 0.9 unchanged unless QLEARN_GAMMA is set. Motivated by a
# prior team's independent finding (Ernst & Striebel 2021, "maverick"): with
# short-horizon features that can't really support long-horizon value
# estimates, a high discount factor pushes the Q-function toward an
# objective the features can't represent, and "do nothing" becomes locally
# optimal -- the same failure shape as our on-target/6-step-commute loop.
GAMMA = float(os.environ.get("QLEARN_GAMMA", "0.9"))
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995  # multiplied in after every round

# Experience replay: online single-step TD(0) update is kept (reacts
# immediately to the current transition), plus a uniform-random replay batch
# each step so learning isn't dominated by whatever happened most recently in
# THIS trajectory. Rare high-magnitude transitions (a self-kill, -10 in one
# step) get resampled many times instead of contributing one gradient step
# and then being forgotten -- see docs/experiments.md, Experiment 6.
REPLAY_CAPACITY = 10_000
REPLAY_BATCH_SIZE = 8
REPLAY_MIN_SIZE = 200

# Target network: bootstrap targets (max_a' Q(s',a')) are computed from a
# periodically-synced copy of the weights, not the weights currently being
# updated. Without this, a self-loop transition (WAIT: the feature vector is
# identical before/after, since the agent's position doesn't change) makes
# replay bootstrap off its own just-updated estimate every time it's
# resampled -- an unstable feedback loop with no counterbalancing signal,
# which is exactly what happened in Experiment 6 before this was added
# (WAIT's Q-value ran away and the agent got stuck idling). Standard DQN
# fix. Synced by *update count*, not by round -- at up to 400 steps/round x
# 9 updates/step, "every N rounds" was tens of thousands of updates between
# syncs, effectively no different from no target network at all (Experiment
# 7). See docs/experiments.md, Experiment 8.
TARGET_SYNC_EVERY_N_UPDATES = 500

# Ablation switch for experiments: with QLEARN_USE_SHAPING=0, only the base
# sparse events below are rewarded (see docs/experiments.md).
USE_SHAPING = os.environ.get("QLEARN_USE_SHAPING", "1") == "1"

# Custom shaping events
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"    # target = nearest coin, else nearest crate-adjacent tile
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
MOVED_TOWARDS_OPPONENT = "MOVED_TOWARDS_OPPONENT"  # symmetric with the above, toward the committed opponent
MOVED_AWAY_FROM_OPPONENT = "MOVED_AWAY_FROM_OPPONENT"
DROPPED_BOMB_SAFE = "DROPPED_BOMB_SAFE"          # bomb dropped with a reachable escape tile
DROPPED_BOMB_UNSAFE = "DROPPED_BOMB_UNSAFE"      # bomb dropped with no escape (likely suicide)
CRATE_PAYOFF_AT_DROP = "CRATE_PAYOFF_AT_DROP"    # paid once per crate a just-dropped bomb will destroy

CRATE_REWARD = 0.3  # per crate; paid at drop time (see _crate_payoff_events), not at destruction

# Rewarded regardless of the shaping switch -- these are the "real" sparse objectives.
#
# KILLED_SELF: 0.0 (was -5.0) -- KILLED_SELF and GOT_KILLED both fire on
# every self-kill (environment.py: KILLED_SELF in the "Kill agents" loop,
# then GOT_KILLED unconditionally in the following "Remove hit agents" loop
# for every agent in agents_hit, self-kills included) -- so a self-kill was
# being charged -5 + -5 = -10, double the intended penalty, since
# Experiment 3. GOT_KILLED alone now carries the full -5.0.
#
# CRATE_DESTROYED: 0.0 (was 0.3) -- moved to CRATE_PAYOFF_AT_DROP, paid
# immediately when the bomb is dropped instead of ~4 steps later when it
# actually detonates (and only if the agent survives to see it). The
# reward for bombing was arriving too late and too conditionally to reach
# the decision that earned it (see docs/experiments.md, Experiment 16).
#
# KILLED_OPPONENT: 2.0 (Experiment 24) -- the real game awards 5 points for
# an opponent kill against 1 for a coin, so weighted 2x COIN_COLLECTED here
# to reflect that relative value without letting a single sparse event
# dominate the reward scale the way a literal 5x would.
BASE_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.COIN_FOUND: 0.2,
    e.CRATE_DESTROYED: 0.0,
    CRATE_PAYOFF_AT_DROP: CRATE_REWARD,
    e.KILLED_SELF: 0.0,
    e.GOT_KILLED: -5.0,
    e.KILLED_OPPONENT: 2.0,
    e.SURVIVED_ROUND: 1.0,
}

# MOVED_INTO_DANGER / MOVED_OUT_OF_DANGER / STAYED_IN_DANGER (flat +-0.3
# events) removed in favor of potential-based escape shaping (Experiment
# 15, see potential() in callbacks.py and its use below) -- Phi(s') -
# Phi(s) added directly to the reward, not routed through this event/lookup
# table, since it's a continuous quantity, not a discrete event.
SHAPING_REWARDS = {
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.1,
    MOVED_TOWARDS_TARGET: 0.1,
    MOVED_AWAY_FROM_TARGET: -0.1,
    MOVED_TOWARDS_OPPONENT: 0.1,
    MOVED_AWAY_FROM_OPPONENT: -0.1,
    DROPPED_BOMB_SAFE: 0.2,
    DROPPED_BOMB_UNSAFE: -0.5,
}

GAME_REWARDS = {**BASE_REWARDS, **(SHAPING_REWARDS if USE_SHAPING else {})}


def setup_training(self):
    self.epsilon = EPSILON_START
    self.round_rewards = []
    self._current_round_reward = 0.0
    self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
    self.target_weights = self.weights.copy()
    self._update_count = 0
    if INSTRUMENT:
        self.bomb_log = []
        self.escape_drops = []
        self.escape_deaths = []


def _instrument_bomb(self, action, old_features, reward):
    if not INSTRUMENT or action != 'BOMB':
        return
    self.bomb_log.append({
        'was_random': bool(getattr(self, '_last_action_was_random', False)),
        'safe_bomb': float(old_features[IDX_SAFE_BOMB]),
        'crate_payoff': float(old_features[IDX_CRATE_PAYOFF]),
        'reward': float(reward),
    })
    with open(INSTRUMENT_FILE, 'w') as f:
        json.dump(self.bomb_log, f)


def _instrument_escape_check(self, action, old_features, game_state, events):
    if not INSTRUMENT:
        return
    is_drop = action == 'BOMB' and e.BOMB_DROPPED in events
    is_death = e.KILLED_SELF in events
    if not (is_drop or is_death):
        return

    round_ = game_state['round']
    step = game_state['step']
    if is_drop:
        self.escape_drops.append({'round': round_, 'step': step,
                                   'safe_bomb': float(old_features[IDX_SAFE_BOMB])})
    if is_death:
        self.escape_deaths.append({'round': round_, 'step': step})
    with open(ESCAPE_CHECK_FILE, 'w') as f:
        json.dump({'drops': self.escape_drops, 'deaths': self.escape_deaths}, f)


def _crate_payoff_events(old_game_state, action, events):
    """CRATE_PAYOFF_AT_DROP once per crate the just-dropped bomb will
    destroy -- unconditional (not gated by USE_SHAPING), replacing
    CRATE_DESTROYED (zeroed in BASE_REWARDS) as the base crate-clearing
    reward, paid at drop time instead of ~4 steps later at detonation.
    Reuses the exact blast-radius computation the crate_payoff *feature*
    uses (get_blast_coords + counting field==1 tiles), not the capped/
    scaled feature value, since we want the true crate count here."""
    if action != 'BOMB' or e.BOMB_DROPPED not in events:
        return []
    field = old_game_state['field']
    pos = old_game_state['self'][3]
    crates_hit = sum(1 for (bx, by) in get_blast_coords(field, pos, s.BOMB_POWER)
                      if field[bx, by] == 1)
    return [CRATE_PAYOFF_AT_DROP] * crates_hit


def _add_shaping_events(old_features, self_action, events):
    """Derive shaping events purely from the already-computed old-state
    feature vector (see state_to_features for the feature layout) and the
    action taken, so no extra BFS/board scans are needed here."""
    if not USE_SHAPING:
        return events

    events = list(events)

    if self_action in MOVE_ACTIONS:
        i = MOVE_ACTIONS.index(self_action)

        # target_dirs is all-zero both when there's no target AND (as of
        # Experiment 15) whenever danger=1 -- state_to_features suppresses
        # it in danger so escape-direction is the only directional pull.
        # Both cases correctly produce neither event below (target_dirs[i]
        # can't be 1.0, and the sum()>0 guard skips MOVED_AWAY_FROM_TARGET
        # too), so no danger-specific handling is needed here.
        target_dirs = old_features[IDX_TARGET_BASE:IDX_TARGET_BASE + 4]
        if target_dirs[i] == 1.0:
            events.append(MOVED_TOWARDS_TARGET)
        elif target_dirs.sum() > 0:
            events.append(MOVED_AWAY_FROM_TARGET)

        # Symmetric with the target shaping above (Experiment 24). Same
        # all-zero-is-ambiguous situation and the same non-issue: all-zero
        # here means either no opponents left, in danger (suppressed, same
        # reasoning as target_dirs), or already adjacent to the committed
        # opponent -- none of which should fire either event, and the
        # sum()>0 guard already skips all three correctly.
        opponent_dirs = old_features[IDX_OPPONENT_DIR_BASE:IDX_OPPONENT_DIR_BASE + 4]
        if opponent_dirs[i] == 1.0:
            events.append(MOVED_TOWARDS_OPPONENT)
        elif opponent_dirs.sum() > 0:
            events.append(MOVED_AWAY_FROM_OPPONENT)

    elif self_action == 'BOMB' and e.BOMB_DROPPED in events:
        events.append(DROPPED_BOMB_SAFE if old_features[IDX_SAFE_BOMB] == 1.0 else DROPPED_BOMB_UNSAFE)

    return events


def _update(self, old_features, action, reward, new_features, terminal):
    action_idx = ACTIONS.index(action)
    q_old = self.weights[action_idx] @ old_features
    if terminal or new_features is None:
        target = reward
    else:
        target = reward + GAMMA * np.max(self.target_weights @ new_features)
    td_error = target - q_old
    self.weights[action_idx] += ALPHA * td_error * old_features

    self._update_count += 1
    if self._update_count % TARGET_SYNC_EVERY_N_UPDATES == 0:
        self.target_weights = self.weights.copy()


def _remember_and_replay(self, old_features, action, reward, new_features, terminal):
    self.replay_buffer.append((old_features, action, reward, new_features, terminal))
    if len(self.replay_buffer) < REPLAY_MIN_SIZE:
        return
    idxs = self.rng.choice(len(self.replay_buffer), size=REPLAY_BATCH_SIZE, replace=False)
    for i in idxs:
        old_f, a, r, new_f, term = self.replay_buffer[i]
        _update(self, old_f, a, r, new_f, term)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    events = _add_shaping_events(old_features, self_action, events)
    events = events + _crate_payoff_events(old_game_state, self_action, events)
    reward = reward_from_events(self, events)
    if USE_SHAPING:
        reward += potential(new_game_state) - potential(old_game_state)
    self._current_round_reward += reward

    _update(self, old_features, self_action, reward, new_features, terminal=False)
    _remember_and_replay(self, old_features, self_action, reward, new_features, terminal=False)
    _instrument_bomb(self, self_action, old_features, reward)
    _instrument_escape_check(self, self_action, old_features, old_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_action is not None:
        last_features = state_to_features(self, last_game_state)
        events = _add_shaping_events(last_features, last_action, events)
        events = events + _crate_payoff_events(last_game_state, last_action, events)
        reward = reward_from_events(self, events)
        if USE_SHAPING:
            reward += 0.0 - potential(last_game_state)  # Phi(terminal) = 0 by convention
        self._current_round_reward += reward
        _update(self, last_features, last_action, reward, None, terminal=True)
        _remember_and_replay(self, last_features, last_action, reward, None, terminal=True)
        _instrument_bomb(self, last_action, last_features, reward)
        _instrument_escape_check(self, last_action, last_features, last_game_state, events)
    else:
        reward = reward_from_events(self, events)
        self._current_round_reward += reward
        if INSTRUMENT and e.KILLED_SELF in events:
            self.escape_deaths.append({'round': last_game_state['round'],
                                        'step': last_game_state['step']})
            with open(ESCAPE_CHECK_FILE, 'w') as f:
                json.dump({'drops': self.escape_drops, 'deaths': self.escape_deaths}, f)

    self.round_rewards.append(self._current_round_reward)
    self._current_round_reward = 0.0
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.current_target = None  # next round reshuffles coins/crates

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.weights, file)


def reward_from_events(self, events: List[str]) -> float:
    reward_sum = sum(GAME_REWARDS.get(event, 0.0) for event in events)
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
