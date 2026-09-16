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
    IDX_TARGET_BASE, IDX_SAFE_BOMB, IDX_OPPONENT_DIR_BASE,
)

# SARSA (on-policy), the second model. Everything except the TD target and
# the deferred-update bookkeeping it requires is copied unchanged from
# qlearn_agent: same 23 features (callbacks.py is byte-for-byte identical),
# same commitment mechanisms, same reward table, same hyperparameters.
#
# The only algorithmic difference: instead of max_a' Q_target(s', a')
# (Q-learning -- value of the *best possible* continuation), SARSA uses
# Q_target(s', a'_actual) -- the value of whatever action the policy will
# actually take next, epsilon-random draws included. See docs/experiments.md,
# Experiment 21 for the hypothesis this is meant to test, written before any
# of this was run.

ALPHA = 0.01
GAMMA = 0.9
# Overridable for curriculum stages (Experiment 28): a warm-started run is
# fine-tuning an already-competent policy, so re-opening at the from-scratch
# exploration rate of 0.3 would spend its first rounds unlearning what it was
# given. Default is unchanged, so every earlier experiment reproduces exactly.
EPSILON_START = float(os.environ.get("QLEARN_EPSILON_START", "0.3"))
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

REPLAY_CAPACITY = 10_000
REPLAY_BATCH_SIZE = 8
REPLAY_MIN_SIZE = 200
TARGET_SYNC_EVERY_N_UPDATES = 500

USE_SHAPING = True

MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
MOVED_TOWARDS_OPPONENT = "MOVED_TOWARDS_OPPONENT"
MOVED_AWAY_FROM_OPPONENT = "MOVED_AWAY_FROM_OPPONENT"
DROPPED_BOMB_SAFE = "DROPPED_BOMB_SAFE"
DROPPED_BOMB_UNSAFE = "DROPPED_BOMB_UNSAFE"
CRATE_PAYOFF_AT_DROP = "CRATE_PAYOFF_AT_DROP"

CRATE_REWARD = 0.3

# KILLED_OPPONENT: 2.0 (Experiment 24) -- see qlearn_agent/train.py for the
# reasoning (real game weights it 5x a coin; 2x here to reflect that without
# letting one sparse event dominate the reward scale).
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
    # Transition (old_features, action, reward, new_features) awaiting the
    # action actually taken from new_features -- SARSA can't compute its
    # target until that's known, one step later than Q-learning can.
    self.pending = None


def _crate_payoff_events(old_game_state, action, events):
    if action != 'BOMB' or e.BOMB_DROPPED not in events:
        return []
    field = old_game_state['field']
    pos = old_game_state['self'][3]
    crates_hit = sum(1 for (bx, by) in get_blast_coords(field, pos, s.BOMB_POWER)
                      if field[bx, by] == 1)
    return [CRATE_PAYOFF_AT_DROP] * crates_hit


def _add_shaping_events(old_features, self_action, events):
    if not USE_SHAPING:
        return events

    events = list(events)

    if self_action in MOVE_ACTIONS:
        i = MOVE_ACTIONS.index(self_action)
        target_dirs = old_features[IDX_TARGET_BASE:IDX_TARGET_BASE + 4]
        if target_dirs[i] == 1.0:
            events.append(MOVED_TOWARDS_TARGET)
        elif target_dirs.sum() > 0:
            events.append(MOVED_AWAY_FROM_TARGET)

        opponent_dirs = old_features[IDX_OPPONENT_DIR_BASE:IDX_OPPONENT_DIR_BASE + 4]
        if opponent_dirs[i] == 1.0:
            events.append(MOVED_TOWARDS_OPPONENT)
        elif opponent_dirs.sum() > 0:
            events.append(MOVED_AWAY_FROM_OPPONENT)

    elif self_action == 'BOMB' and e.BOMB_DROPPED in events:
        events.append(DROPPED_BOMB_SAFE if old_features[IDX_SAFE_BOMB] == 1.0 else DROPPED_BOMB_UNSAFE)

    return events


def _sarsa_update(self, old_features, action, reward, new_features, next_action, terminal):
    """Q(s,a) <- Q(s,a) + alpha*(r + gamma*Q_target(s', a') - Q(s,a)), where
    a' is the action *actually* taken from s' -- not argmax. terminal (or
    next_action is None) means no bootstrap: target = reward only."""
    action_idx = ACTIONS.index(action)
    q_old = self.weights[action_idx] @ old_features
    if terminal or new_features is None or next_action is None:
        target = reward
    else:
        next_action_idx = ACTIONS.index(next_action)
        target = reward + GAMMA * (self.target_weights[next_action_idx] @ new_features)
    td_error = target - q_old
    self.weights[action_idx] += ALPHA * td_error * old_features

    self._update_count += 1
    if self._update_count % TARGET_SYNC_EVERY_N_UPDATES == 0:
        self.target_weights = self.weights.copy()


def _remember_and_replay(self, old_features, action, reward, new_features, next_action, terminal):
    self.replay_buffer.append((old_features, action, reward, new_features, next_action, terminal))
    if len(self.replay_buffer) < REPLAY_MIN_SIZE:
        return
    idxs = self.rng.choice(len(self.replay_buffer), size=REPLAY_BATCH_SIZE, replace=False)
    for i in idxs:
        old_f, a, r, new_f, next_a, term = self.replay_buffer[i]
        _sarsa_update(self, old_f, a, r, new_f, next_a, term)


def _flush_pending(self, next_action):
    """Finalize the pending transition now that the action actually taken
    from its next-state is known. Called at the *start* of the following
    game_events_occurred/end_of_round, before that call's own transition
    becomes the new pending one -- next_action here is exactly that call's
    self_action/last_action parameter, i.e. the real action the policy took,
    never recomputed or assumed."""
    if self.pending is None:
        return
    old_features, action, reward, new_features = self.pending
    self.logger.debug(f"SARSA flush: pending_action={action} next_action={next_action}")
    _sarsa_update(self, old_features, action, reward, new_features, next_action, terminal=False)
    _remember_and_replay(self, old_features, action, reward, new_features, next_action, terminal=False)
    self.pending = None


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return

    # self_action is the action taken FROM old_game_state -- exactly the
    # "next action" needed to finalize whatever transition is still pending
    # from the previous call. Must flush before this step's own transition
    # overwrites self.pending.
    _flush_pending(self, self_action)

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    events = _add_shaping_events(old_features, self_action, events)
    events = events + _crate_payoff_events(old_game_state, self_action, events)
    reward = reward_from_events(self, events)
    if USE_SHAPING:
        reward += potential(new_game_state) - potential(old_game_state)
    self._current_round_reward += reward

    self.pending = (old_features, self_action, reward, new_features)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_action is not None:
        # last_action was taken FROM last_game_state, which is exactly the
        # new_game_state of whatever transition is still pending -- flush it
        # non-terminally (it's a real transition with a real next action,
        # even though this is the last step of the round).
        _flush_pending(self, last_action)

        last_features = state_to_features(self, last_game_state)
        events = _add_shaping_events(last_features, last_action, events)
        events = events + _crate_payoff_events(last_game_state, last_action, events)
        reward = reward_from_events(self, events)
        if USE_SHAPING:
            reward += 0.0 - potential(last_game_state)
        self._current_round_reward += reward

        # This step's own transition IS terminal: no next action exists.
        _sarsa_update(self, last_features, last_action, reward, None, None, terminal=True)
        _remember_and_replay(self, last_features, last_action, reward, None, None, terminal=True)
    else:
        # No new transition this call (last_action is None) -- flush
        # whatever was pending as terminal too, since no further action was
        # ever observed for it.
        if self.pending is not None:
            old_features, action, pending_reward, new_features = self.pending
            _sarsa_update(self, old_features, action, pending_reward, None, None, terminal=True)
            _remember_and_replay(self, old_features, action, pending_reward, None, None, terminal=True)
            self.pending = None
        reward = reward_from_events(self, events)
        self._current_round_reward += reward

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
