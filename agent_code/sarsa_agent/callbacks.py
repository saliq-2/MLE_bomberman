import os
import pickle
from collections import deque

import numpy as np

import settings as s

# Task 2 introduces crates, so bombing is needed again.
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# (dx, dy) for each movement action, matching environment.py's perform_agent_action
DIRECTIONS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}

# bias + 4 target-direction one-hot + 4 valid-move + 1 in-danger
# + 4 safe-move + 1 safe-to-bomb + 1 crate-payoff + 1 crate-target-distance
#
# No separate "bomb available" feature: it's perfectly collinear with
# safe-to-bomb whenever bombing is actually viable (safe-to-bomb can only be
# 1 if a bomb is available), and having both let the model dump a large
# negative weight onto the always-co-occurring "bomb available" term,
# making Q(BOMB) deeply negative even when safe-to-bomb correctly fired --
# net effect, the agent never bombed at all (see docs/experiments.md,
# Experiment 10). INVALID_ACTION already penalizes trying to bomb with none
# left, so bomb-availability doesn't need its own feature.
#
# crate-payoff and crate-target-distance added for the same reason
# (Experiment 11): "safe to bomb" only says a bomb here wouldn't kill you,
# it says nothing about whether bombing here is *worthwhile* -- so BOMB's
# weights had nothing to learn a positive association from besides sparse,
# delayed CRATE_DESTROYED/COIN_FOUND events. crate-payoff gives an immediate,
# dense signal ("how many crates would this bomb actually hit") the model
# can attach a per-step reward to well before those delayed events land.
CRATE_PAYOFF_CAP = 4     # crates hit by one bomb, above which we stop counting
CRATE_DIST_CAP = 10      # BFS steps to nearest crate-adjacent tile, capped
OPPONENT_DIST_CAP = 15   # BFS steps to the committed opponent, capped (Experiment 24)
#
# crate_payoff * safe_bomb, added as its own dimension (Experiment 12):
# crate-dense tiles are also the tiles with the fewest escape routes, so
# crate_payoff and "unsafe to bomb" were positively correlated in training
# data dominated by early random exploration -- a purely additive linear
# model can't represent "good if high-payoff AND safe", only the marginal
# (confounded) association, and it learned crate_payoff's weight as
# negative for BOMB as a result (see docs/experiments.md, Experiment 11).
# bare crate_payoff is kept alongside this so the two runs stay comparable.
#
# on_target, added as its own dimension (Experiment 13): target_onehot goes
# all-zero in two semantically different situations -- "no target exists"
# and "you're standing exactly on your target" -- and the model had no way
# to tell them apart. That ambiguity, combined with the committed target
# never being invalidated on arrival (see _select_target), produced a
# sustained on-target/off-target 2-cycle: confirmed directly by dumping a
# real action sequence (docs/experiments.md, Experiment 11) -- pos and
# target_onehot alternate in lockstep for an entire 400-step round, not a
# rare edge case. on_target is computed from the *prior* commitment, before
# _select_target invalidates and re-picks (see state_to_features).
#
# escape-direction one-hot, added as its own dimension (Experiment 18): an
# epsilon sweep at eval showed the "never bomb" greedy policy is not a
# calibrated risk-averse preference -- the moment any exploration noise let
# bombing happen at all, survival collapsed (55-96% suicide even at
# epsilon=0.02-0.10), meaning escape *execution*, not the drop decision, is
# the real bottleneck. Before this, "which of the 4 directions is safe"
# (IDX_SAFE_BASE) could light up multiple directions at once with no single
# best answer, and nothing committed the agent to one -- the same
# structural gap current_target/on_target closed for navigation, now
# applied to escape: commit to one escape direction (via the same
# time-expanded safety search as _can_escape_own_bomb) and keep it until
# it stops being safe or danger clears, instead of re-deriving from
# scratch, ambiguously, every step (see _select_escape_direction).
#
# Opponent-aware features, added as a group (Experiment 24): Experiment 23
# showed both agents are completely opponent-blind (no feature ever reads
# game_state['others'] for anything but static-obstacle purposes), and in
# the tournament config (3 simultaneous opponents) deaths-by-opponent
# (101.2/round) had overtaken self-kills (75.4/round) as qlearn_agent's
# dominant failure mode. Four additions:
#   - opponent_dist: BFS distance to the committed opponent target, capped/
#     scaled, same style as crate_dist.
#   - opponent-direction one-hot (4 dims): first-step direction toward the
#     committed opponent's *current* position, same commit-and-hold pattern
#     as current_target (see _select_opponent_target) -- committed by
#     NAME, not position, since (unlike coins/crates) opponents move every
#     step; re-picking "nearest" fresh each step would reproduce the exact
#     flip-flop failure from Experiments 1c/13, just against a moving
#     target instead of a static one. Suppressed to all-zero in danger,
#     same as the coin/crate target direction (Experiment 15) and for the
#     same reason: escape should be the only directional pull while a bomb
#     is ticking.
#   - opponent_in_blast: would a bomb dropped right now hit any opponent's
#     *current* tile -- reuses get_blast_coords, the same computation
#     crate_payoff already uses.
#   - opponent_trapped: would the committed opponent have no escape from a
#     bomb dropped at our position -- reuses _can_escape_own_bomb's
#     time-expanded safety search, generalized to take a separate
#     `start_pos` (the escaping agent's position) distinct from the bomb's
#     origin, so the same correctness work from Experiment 13/14 (per-step
#     danger timing, explosion residue, other active bombs) applies to
#     "can *they* escape *our* bomb", not just "can we escape our own".
#
# Escape-robustness features, added as a group (Experiment 26): IDX_SAFE_BOMB
# answers "does *a* route out exist", which is the right question in a solo
# game and the wrong one with three other bombers on the board -- a
# single-route escape holds only until someone drops a bomb across that one
# corridor while ours is still ticking. Experiment 25 measured how often
# that situation actually arises (bucket (a): an opponent bomb dropped
# inside our own drop-to-death window) at 66.3% of the submitted sarsa
# baseline's self-kills. What Experiment 25 ruled out was the narrower
# claim that bucket (a) explained the *increase* in qlearn's self-kill rate
# under opponent training -- not that bucket (a) is irreducible in absolute
# terms, which is what these features go after.
#
# Two dimensions rather than one, because the obvious single answer is too
# sparse to learn from: "two or more distinct escape corridors" fires in
# only 1.6% of randomly sampled classic-density states (measured before
# training anything, scripts/check_escape_equivalence.py) -- a feature that
# is off 98.4% of the time gives the weight vector almost nothing to attach
# to. So the binary redundancy signal is paired with a graded one, escape
# slack, which is defined everywhere a bomb is droppable at all and
# degrades smoothly instead of switching. Added as a group for the same
# reason Experiment 24 added its seven opponent features as a group; the
# per-dimension attribution that buys is traded away knowingly, and the
# ablation that would recover it is listed in "Next steps".
N_FEATURES = 32

# Named indices into the feature vector state_to_features returns, so
# train.py's shaping/instrumentation code doesn't hardcode magic numbers
# that can silently drift out of sync with this layout when it changes
# (audited once by hand for the current layout -- see docs/experiments.md;
# these constants exist so the next feature-set change doesn't need that
# audit repeated).
IDX_BIAS = 0
IDX_TARGET_BASE = 1    # + MOVE_ACTIONS.index(action), 4 slots: 1-4
IDX_VALID_BASE = 5     # + MOVE_ACTIONS.index(action), 4 slots: 5-8
IDX_DANGER = 9
IDX_SAFE_BASE = 10     # + MOVE_ACTIONS.index(action), 4 slots: 10-13
IDX_SAFE_BOMB = 14
IDX_CRATE_PAYOFF = 15
IDX_CRATE_DIST = 16
IDX_PAYOFF_X_SAFE = 17
IDX_ON_TARGET = 18
IDX_ESCAPE_BASE = 19   # + MOVE_ACTIONS.index(action), 4 slots: 19-22
IDX_OPPONENT_DIST = 23
IDX_OPPONENT_DIR_BASE = 24  # + MOVE_ACTIONS.index(action), 4 slots: 24-27
IDX_OPPONENT_IN_BLAST = 28
IDX_OPPONENT_TRAPPED = 29
IDX_SAFE_BOMB_ROBUST = 30
IDX_ESCAPE_SLACK = 31

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model.pt')

# Guided exploration (Experiment 20, off by default): idea prompted by
# reading a prior team's report (Ernst & Striebel 2021, "maverick" -- read
# for strategy only, no code touched, professor's permission for this) --
# they mixed a heuristic policy into their epsilon-random fraction instead
# of pure uniform random, and reported it sped up convergence substantially.
# Applied narrowly here: while in danger, uniform-random exploration means
# most exploratory steps walk straight into the blast, which is exactly the
# training-data skew diagnosed in Experiment 10/11 (bad bombing outcomes
# dominate the replay buffer). With this on, an epsilon-random step in
# danger follows the already-computed committed escape direction with
# probability GUIDED_EXPLORE_ESCAPE_PROB instead of picking uniformly.
# Never used outside epsilon-random exploration and never overrides the
# learned greedy policy -- self.train=False (eval/tournament) is unaffected
# regardless of this flag, since epsilon=0 there means was_random is never
# True in the first place.
GUIDED_EXPLORE = os.environ.get("QLEARN_GUIDED_EXPLORE", "0") == "1"
GUIDED_EXPLORE_ESCAPE_PROB = 0.7


def setup(self):
    # Agent-side RNG, independent of --seed (which only controls the world's
    # board layout -- see docs/experiments.md). Every random draw the agent
    # makes (epsilon-greedy, replay sampling in train.py) goes through this,
    # so a run is fully reproducible given (board seed, QLEARN_SEED).
    seed = int(os.environ.get("QLEARN_SEED", "0"))
    self.rng = np.random.default_rng(seed)
    self.logger.info(f"Using QLEARN_SEED={seed}")

    # Warm start for curriculum training (Experiment 28). Training otherwise
    # always begins from zero weights, which means every opponent-training run
    # in this log so far (Experiment 24's included) had to learn navigation,
    # bomb safety and opponent play simultaneously inside 1000 rounds, from
    # nothing. Pointing QLEARN_INIT_FROM at an existing checkpoint starts the
    # run from a policy that already solves Task 2 and lets the opponent
    # rounds do only the work they are actually for.
    #
    # Only consulted while training, so evaluation and tournament play are
    # bit-for-bit unaffected whether or not the variable is set. A width
    # mismatch is a hard error rather than a silent reshape: loading a
    # narrower checkpoint would misalign every feature index, and the
    # zero-padding that makes widths compatible belongs in
    # scripts/pad_checkpoints.py, where it is verified, not here.
    init_from = os.environ.get("QLEARN_INIT_FROM", "")
    if self.train and init_from:
        # The framework runs agent callbacks in a separate process whose
        # working directory is not the repo root, so a relative path handed in
        # through the environment does not resolve the way it would in the
        # shell that set it. Resolve against the repo root (two levels up from
        # this file) rather than against the caller's cwd.
        if not os.path.isabs(init_from):
            init_from = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), init_from)
        with open(init_from, "rb") as file:
            warm = pickle.load(file)
        if warm.shape != (len(ACTIONS), N_FEATURES):
            raise ValueError(
                f"QLEARN_INIT_FROM checkpoint has shape {warm.shape}, "
                f"expected {(len(ACTIONS), N_FEATURES)} -- pad it first "
                f"(scripts/pad_checkpoints.py)")
        self.logger.info(f"Warm-starting from {init_from}")
        self.weights = warm.copy()
    elif self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up fresh linear Q-model.")
        self.weights = np.zeros((len(ACTIONS), N_FEATURES))
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.weights = pickle.load(file)
    # Committed navigation target (see docs/experiments.md, Experiment 1c):
    # picking "nearest coin" fresh every step lets the identity of "nearest"
    # flip between two similarly-placed coins as the agent moves, causing a
    # stable back-and-forth oscillation under a pure greedy policy. Instead
    # we commit to one target and path to that specific point until it's
    # reached or invalidated.
    self.current_target = None
    # Committed escape direction (Experiment 18) -- same reasoning as
    # current_target above, applied to escaping danger instead of navigation.
    self.escape_direction = None
    # Committed opponent target, by name not position (Experiment 24) --
    # opponents move every step, unlike coins/crates, so "position" can't
    # be the commitment key; the opponent's name can.
    self.opponent_target_name = None


def act(self, game_state: dict) -> str:
    if game_state['step'] == 1:
        self.current_target = None  # fresh round -> coins/crates reshuffled
        self.escape_direction = None
        self.opponent_target_name = None

    features = state_to_features(self, game_state)
    pos = game_state['self'][3]

    # Eval defaults to pure greedy (epsilon=0), matching the tournament's
    # self.train=False. QLEARN_EVAL_EPSILON overrides this for diagnostic
    # sweeps only -- unset, behavior is unchanged.
    epsilon = self.epsilon if self.train else float(os.environ.get("QLEARN_EVAL_EPSILON", "0.0"))
    was_random = self.rng.random() < epsilon
    self._last_action_was_random = was_random  # read by train.py's instrumentation

    if was_random:
        in_danger = features[IDX_DANGER] == 1.0
        if GUIDED_EXPLORE and in_danger and self.escape_direction is not None \
                and self.rng.random() < GUIDED_EXPLORE_ESCAPE_PROB:
            action = self.escape_direction
            self.logger.debug("Guided exploration: following committed escape direction.")
        else:
            action = self.rng.choice(ACTIONS)
            self.logger.debug("Choosing action purely at random.")
    else:
        q_values = self.weights @ features
        action = ACTIONS[int(np.argmax(q_values))]
        self.logger.debug(f"Q-values {dict(zip(ACTIONS, q_values))}, choosing {action}")

    self.logger.debug(f"trace step={game_state['step']} pos={pos} "
                       f"target_onehot={features[IDX_TARGET_BASE:IDX_TARGET_BASE+4].tolist()} "
                       f"on_target={features[IDX_ON_TARGET]:.0f} action={action}")
    return action


def _occupied_tiles(bombs, others):
    occupied = {pos for pos, _ in bombs}
    occupied.update(pos for (_, _, _, pos) in others)
    return occupied


def _is_free(field, occupied, pos):
    x, y = pos
    if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
        return False
    return field[x, y] == 0 and pos not in occupied


def _bfs_first_step(field, occupied, start, is_goal):
    """BFS over free tiles; returns (first_step_direction, goal_pos) for the
    nearest tile for which is_goal(pos) is True, or (None, None) if
    unreachable."""
    visited = {start}
    queue = deque([(start, None)])

    while queue:
        pos, first_step = queue.popleft()
        if first_step is not None and is_goal(pos):
            return first_step, pos

        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or not _is_free(field, occupied, nxt):
                continue
            visited.add(nxt)
            queue.append((nxt, first_step if first_step is not None else action))

    return None, None


def _bfs_to_point(field, occupied, start, target):
    """BFS over free tiles; returns the first-step direction on a shortest
    path from start to the specific tile `target` (not "nearest of many" --
    this is what keeps target-following monotonic/stable, see setup())."""
    if start == target:
        return None
    visited = {start}
    queue = deque([(start, None)])

    while queue:
        pos, first_step = queue.popleft()
        if pos == target:
            return first_step

        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or not _is_free(field, occupied, nxt):
                continue
            visited.add(nxt)
            queue.append((nxt, first_step if first_step is not None else action))

    return None


def _bfs_distance(field, occupied, start, is_goal):
    """BFS over free tiles; returns the number of steps to the nearest tile
    for which is_goal(pos) is True (0 if `start` itself qualifies), or None
    if unreachable."""
    if is_goal(start):
        return 0
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        pos, dist = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or not _is_free(field, occupied, nxt):
                continue
            if is_goal(nxt):
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def get_blast_coords(field, pos, power):
    """Same rule as items.py Bomb.get_blast_coords: blast travels through
    crates but is stopped by stone walls."""
    x, y = pos
    blast = [(x, y)]
    for dx, dy in DIRECTIONS.values():
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            blast.append((nx, ny))
    return blast


def _danger_zone(field, bombs, explosion_map):
    """Tiles that are either actively exploding or in the blast line of a
    still-ticking bomb (conservative: ignores the countdown, since a ticking
    bomb's line is unsafe to linger on regardless of how many steps are left)."""
    danger = {(x, y) for x in range(field.shape[0]) for y in range(field.shape[1])
              if explosion_map[x, y] > 0}
    for pos, _ in bombs:
        danger.update(get_blast_coords(field, pos, s.BOMB_POWER))
    return danger


def _bomb_danger_windows(field, bombs):
    """For each bomb (pos, countdown), the blast tiles and the *relative*
    time window (steps from now) during which they're dangerous.

    Timing, traced from environment.py: a bomb with countdown c detonates
    exactly c steps from now (do_step -> update_bombs decrements after
    each step, matching game_state's documented "0 = about to explode").
    On detonation it's dangerous for the step it explodes plus one more
    residue step before turning to harmless smoke (update_explosions:
    timer starts at EXPLOSION_TIMER, next_stage/harmless once it hits 0 --
    2 dangerous ticks for EXPLOSION_TIMER=2, i.e. window [c, c+1])."""
    windows = []
    for bpos, countdown in bombs:
        blast = frozenset(get_blast_coords(field, bpos, s.BOMB_POWER))
        windows.append((blast, countdown, countdown + (s.EXPLOSION_TIMER - 1)))
    return windows


def _dangerous_at(pos, t, windows):
    return any(pos in blast and start <= t <= end for blast, start, end in windows)


def _can_escape_own_bomb(field, occupied, pos, bombs, start_pos=None):
    """Would a bomb dropped at `pos` right now leave a valid escape, for an
    agent starting at `start_pos` (defaults to `pos`, i.e. our own escape)?

    `start_pos` != `pos` answers a different question (Experiment 24): can
    *someone else*, currently at `start_pos`, escape a bomb *we* drop at
    `pos`? Used for the opponent_trapped feature -- same search, just a
    different starting point for the BFS than the bomb's origin.

    A tile only counts as valid if: it is outside the blast of the bomb
    being dropped; it is reachable within BOMB_TIMER steps; every tile on
    the path to it is itself safe at the exact time-step the agent would
    occupy it (so a path is rejected if it would walk through another
    bomb's blast while that bomb is actually dangerous -- pre-detonation
    tiles of the *new* bomb are not a hazard yet, but another already-
    ticking bomb's line can be); and the resting tile stays safe through
    the new bomb's own residue round (see docs/experiments.md, Experiment
    14 -- this replaces the earlier static single-bomb BFS, which ignored
    any other active bombs entirely and only checked the stepping-distance
    budget, not per-step timing).

    The relevant horizon is fixed to *this* bomb's own resolution window
    (BOMB_TIMER + residue), not extended to cover other bombs' windows
    however far in the future -- a bomb with a countdown far beyond that
    has nothing to do with whether dropping now is safe; only checked for
    overlap with the near-term horizon, same as any other bomb."""
    if start_pos is None:
        start_pos = pos
    hypothetical = list(bombs) + [(pos, s.BOMB_TIMER)]
    windows = _bomb_danger_windows(field, hypothetical)
    horizon = s.BOMB_TIMER + (s.EXPLOSION_TIMER - 1)

    def safe_to_rest(p, t_reach):
        return all(not _dangerous_at(p, t, windows) for t in range(t_reach, horizon + 1))

    visited = {(start_pos, 0)}
    queue = deque([(start_pos, 0)])
    while queue:
        cur, t = queue.popleft()
        if t > 0 and safe_to_rest(cur, t):
            return True
        if t == s.BOMB_TIMER:
            continue
        for dx, dy in DIRECTIONS.values():
            nxt = (cur[0] + dx, cur[1] + dy)
            nt = t + 1
            if (nxt, nt) in visited or not _is_free(field, occupied, nxt):
                continue
            if _dangerous_at(nxt, nt, windows):
                continue
            visited.add((nxt, nt))
            queue.append((nxt, nt))
    return False


def _escape_options(field, occupied, pos, bombs):
    """If a bomb were dropped at `pos` right now: which first moves still
    lead to a genuine escape, and how early can safety be reached?

    Returns `(first_steps, earliest_t)` -- the set of action names that work
    as an opening move, and the smallest number of steps after which we
    could be standing somewhere that stays safe through the whole blast,
    or None if there is no escape at all.

    The search itself is the same one _can_escape_own_bomb runs: same
    hypothetical bomb, same per-step danger windows, same horizon, same
    rule that the resting tile must survive the residue round. The only
    change is that it is run once per opening direction and reports which
    ones succeeded, instead of collapsing everything to a single bool. A
    path may wander freely after its first step; the commitment being
    measured is only to that first move, which is exactly the question the
    redundancy feature asks (if this corridor is taken from me, is there a
    second one?).

    Kept deliberately separate from _can_escape_own_bomb rather than folded
    into it. That function feeds features 0-29, and Experiment 25's
    zero-padding equivalence argument for every pre-Experiment-26
    checkpoint rests on the computation of those features being
    bit-identical -- so it is left untouched, and the two are checked
    against each other (len(first_steps) >= 1 must equal
    _can_escape_own_bomb) in scripts/check_escape_equivalence.py instead of
    the agreement being assumed.
    """
    hypothetical = list(bombs) + [(pos, s.BOMB_TIMER)]
    windows = _bomb_danger_windows(field, hypothetical)
    horizon = s.BOMB_TIMER + (s.EXPLOSION_TIMER - 1)

    def safe_to_rest(p, t_reach):
        return all(not _dangerous_at(p, t, windows) for t in range(t_reach, horizon + 1))

    first_steps = set()
    earliest_t = None

    for action, (dx, dy) in DIRECTIONS.items():
        first = (pos[0] + dx, pos[1] + dy)
        if not _is_free(field, occupied, first) or _dangerous_at(first, 1, windows):
            continue
        visited = {(first, 1)}
        queue = deque([(first, 1)])
        while queue:
            # Uniform step cost, so the queue pops in non-decreasing t and
            # the first resting tile found down this branch is its earliest.
            cur, t = queue.popleft()
            if safe_to_rest(cur, t):
                first_steps.add(action)
                earliest_t = t if earliest_t is None else min(earliest_t, t)
                break
            if t == s.BOMB_TIMER:
                continue
            for ddx, ddy in DIRECTIONS.values():
                nxt = (cur[0] + ddx, cur[1] + ddy)
                nt = t + 1
                if (nxt, nt) in visited or not _is_free(field, occupied, nxt):
                    continue
                if _dangerous_at(nxt, nt, windows):
                    continue
                visited.add((nxt, nt))
                queue.append((nxt, nt))

    return first_steps, earliest_t


def _distance_to_safety(field, occupied, pos, bombs):
    """BFS distance (steps) from pos to the nearest tile safe under the
    CURRENT bomb configuration (no hypothetical new bomb) -- reuses the
    same time-expanded safety logic as _can_escape_own_bomb. Returns 0 if
    pos is already safe, or horizon+1 (a fixed worst-case value, not
    unbounded) if no safe tile is reachable within the horizon."""
    windows = _bomb_danger_windows(field, bombs)
    if not windows:
        return 0
    horizon = max(end for _, _, end in windows)

    def safe_to_rest(p, t_reach):
        return all(not _dangerous_at(p, t, windows) for t in range(t_reach, horizon + 1))

    if safe_to_rest(pos, 0):
        return 0

    visited = {(pos, 0)}
    queue = deque([(pos, 0)])
    while queue:
        cur, t = queue.popleft()
        if t > 0 and safe_to_rest(cur, t):
            return t
        if t == horizon:
            continue
        for dx, dy in DIRECTIONS.values():
            nxt = (cur[0] + dx, cur[1] + dy)
            nt = t + 1
            if (nxt, nt) in visited or not _is_free(field, occupied, nxt):
                continue
            if _dangerous_at(nxt, nt, windows):
                continue
            visited.add((nxt, nt))
            queue.append((nxt, nt))
    return horizon + 1


def _escape_first_step(field, occupied, pos, bombs):
    """First-step direction toward the nearest tile safe under the current
    bomb configuration (time-expanded, same danger-window logic as
    _distance_to_safety), or None if pos is already safe or no safe tile is
    reachable within the horizon."""
    windows = _bomb_danger_windows(field, bombs)
    if not windows:
        return None
    horizon = max(end for _, _, end in windows)

    def safe_to_rest(p, t_reach):
        return all(not _dangerous_at(p, t, windows) for t in range(t_reach, horizon + 1))

    if safe_to_rest(pos, 0):
        return None

    visited = {(pos, 0)}
    queue = deque([(pos, 0, None)])
    while queue:
        cur, t, first_step = queue.popleft()
        if t > 0 and safe_to_rest(cur, t):
            return first_step
        if t == horizon:
            continue
        for action, (dx, dy) in DIRECTIONS.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            nt = t + 1
            if (nxt, nt) in visited or not _is_free(field, occupied, nxt):
                continue
            if _dangerous_at(nxt, nt, windows):
                continue
            visited.add((nxt, nt))
            queue.append((nxt, nt, first_step if first_step is not None else action))
    return None


def _direction_still_safe(field, occupied, pos, bombs, action):
    """Is taking `action` right now still a valid, immediately-safe step?
    Used to decide whether a committed escape direction can be kept as-is
    (see _select_escape_direction)."""
    if action not in DIRECTIONS:
        return False
    dx, dy = DIRECTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not _is_free(field, occupied, nxt):
        return False
    windows = _bomb_danger_windows(field, bombs)
    return not _dangerous_at(nxt, 1, windows)


def _select_escape_direction(self, field, occupied, pos, bombs, in_danger):
    """Commit to one escape direction and keep it until it stops being
    safe or danger clears, instead of re-deriving "which directions are
    safe" fresh and ambiguously every step (see N_FEATURES docstring,
    Experiment 18)."""
    if not in_danger:
        self.escape_direction = None
        return None
    if self.escape_direction is not None and \
            _direction_still_safe(field, occupied, pos, bombs, self.escape_direction):
        return self.escape_direction
    self.escape_direction = _escape_first_step(field, occupied, pos, bombs)
    return self.escape_direction


POTENTIAL_SCALE = -0.5


def potential(game_state):
    """Potential-based escape shaping (Experiment 15): 0 when not in
    danger, else -0.5 x (BFS distance to the nearest tile safe under the
    current bomb configuration). Used as Phi(s'} - Phi(s) added to the
    reward every transition, replacing the flat MOVED_INTO_DANGER /
    MOVED_OUT_OF_DANGER / STAYED_IN_DANGER events (see docs/experiments.md,
    Experiment 15)."""
    if game_state is None:
        return 0.0

    field = game_state['field']
    pos = game_state['self'][3]
    bombs = game_state['bombs']
    others = game_state['others']
    explosion_map = game_state['explosion_map']
    occupied = _occupied_tiles(bombs, others)

    if pos not in _danger_zone(field, bombs, explosion_map):
        return 0.0

    return POTENTIAL_SCALE * _distance_to_safety(field, occupied, pos, bombs)


def _is_crate_adjacent(field, pos):
    x, y = pos
    return any(field[x + dx, y + dy] == 1 for dx, dy in DIRECTIONS.values()
               if 0 <= x + dx < field.shape[0] and 0 <= y + dy < field.shape[1])


def _select_target(self, field, occupied, pos, coins):
    """Commit to a single target tile and keep it until reached/invalidated,
    instead of re-picking "nearest" every step (see setup()).

    Reaching the target invalidates it (Experiment 13): otherwise, once
    pos == current_target, the target-direction one-hot goes permanently
    all-zero with no further pull, and if the policy doesn't bomb on that
    exact step, nothing about the committed-target state changes as it
    wanders off and back -- a sustained on-target/off-target 2-cycle,
    confirmed directly in Experiment 11's action-sequence dump. Forcing a
    fresh pick every time the agent is standing on its target keeps the
    feature vector live instead of frozen. Whether pos was the *prior*
    target (for the on_target feature) is captured by the caller before
    this runs, since target gets cleared here."""
    target = self.current_target
    if target == pos:
        target = None

    if coins:
        if target not in coins:
            _, target = _bfs_first_step(field, occupied, pos, lambda p: p in set(coins))
    else:
        if target is None or not (target == pos or _is_free(field, occupied, target)) \
                or not _is_crate_adjacent(field, target):
            _, target = _bfs_first_step(field, occupied, pos, lambda p: _is_crate_adjacent(field, p))

    self.current_target = target
    return target


def _select_opponent_target(self, pos, others):
    """Commit to one opponent, by name, and hold it until it's no longer
    present (dead/eliminated) -- not by position, since opponents move
    every step (unlike coins/crates), so re-deriving "nearest" fresh each
    step would reproduce the exact flip-flop failure diagnosed in
    Experiments 1c/13, just against a moving target. Returns the committed
    opponent's *current* position (or None if no opponents remain).

    Nearest-of-many at commit time is picked by Manhattan distance, a
    cheap, stable tie-break -- BFS-exact distance isn't needed just to
    decide *which* opponent to commit to; the direction/distance features
    computed against the committed target afterward use a real BFS."""
    by_name = {o[0]: o[3] for o in others}
    if self.opponent_target_name not in by_name:
        self.opponent_target_name = None
        if by_name:
            self.opponent_target_name = min(
                by_name, key=lambda name: abs(by_name[name][0] - pos[0]) + abs(by_name[name][1] - pos[1]))
    return by_name.get(self.opponent_target_name)


def state_to_features(self, game_state: dict) -> np.array:
    if game_state is None:
        return np.zeros(N_FEATURES)

    field = game_state['field']
    pos = game_state['self'][3]
    bombs_left = game_state['self'][2]
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = game_state['others']
    explosion_map = game_state['explosion_map']

    occupied = _occupied_tiles(bombs, others)

    features = np.zeros(N_FEATURES)
    features[IDX_BIAS] = 1.0

    prior_target = self.current_target
    features[IDX_ON_TARGET] = 1.0 if pos == prior_target else 0.0

    danger = _danger_zone(field, bombs, explosion_map)
    features[IDX_DANGER] = 1.0 if pos in danger else 0.0

    # Target commitment still updates every step regardless of danger (so it
    # picks back up correctly once danger clears), but the target-direction
    # one-hot itself is suppressed while in danger (Experiment 15): the
    # escape-direction features (IDX_SAFE_BASE) should be the only
    # directional pull while a bomb is ticking, not competing with "go
    # toward the crate/coin" -- see docs/experiments.md.
    target = _select_target(self, field, occupied, pos, coins)
    target_direction = _bfs_to_point(field, occupied, pos, target) if target is not None else None

    if features[IDX_DANGER] == 0.0:
        for i, action in enumerate(MOVE_ACTIONS):
            if action == target_direction:
                features[IDX_TARGET_BASE + i] = 1.0

    for i, (action, (dx, dy)) in enumerate(DIRECTIONS.items()):
        nxt = (pos[0] + dx, pos[1] + dy)
        valid = _is_free(field, occupied, nxt)
        features[IDX_VALID_BASE + i] = 1.0 if valid else 0.0
        features[IDX_SAFE_BASE + i] = 1.0 if valid and nxt not in danger else 0.0

    # Committed escape direction (Experiment 18): unlike IDX_SAFE_BASE above
    # (which can light up several directions at once with no single best
    # answer), this commits to exactly one and holds it until it stops
    # being safe or danger clears -- see _select_escape_direction.
    escape_direction = _select_escape_direction(self, field, occupied, pos, bombs,
                                                 in_danger=features[IDX_DANGER] == 1.0)
    for i, action in enumerate(MOVE_ACTIONS):
        if action == escape_direction:
            features[IDX_ESCAPE_BASE + i] = 1.0

    features[IDX_SAFE_BOMB] = 1.0 if (bombs_left and _can_escape_own_bomb(field, occupied, pos, bombs)) else 0.0

    # Experiment 26. Both gated on bombs_left for the same reason
    # IDX_SAFE_BOMB is: with no bomb in hand the question is hypothetical,
    # so they read 0 and INVALID_ACTION stays the thing that teaches "don't
    # try to bomb with none left" (Experiment 10).
    if bombs_left:
        _first_steps, _earliest_t = _escape_options(field, occupied, pos, bombs)
        features[IDX_SAFE_BOMB_ROBUST] = 1.0 if len(_first_steps) >= 2 else 0.0
        # Slack, scaled to [0, 1]: 1 would be escaping instantly, 0 is
        # reaching safety only on the step the bomb goes off. Reads 0 when
        # there is no escape at all, which coincides with the worst
        # survivable case rather than being distinguishable from it -- that
        # distinction is already carried by IDX_SAFE_BOMB, so this dimension
        # does not need to repeat it.
        if _earliest_t is not None:
            features[IDX_ESCAPE_SLACK] = max(
                0.0, (s.BOMB_TIMER - _earliest_t) / s.BOMB_TIMER)

    crates_hit = sum(1 for (bx, by) in get_blast_coords(field, pos, s.BOMB_POWER) if field[bx, by] == 1)
    features[IDX_CRATE_PAYOFF] = min(crates_hit, CRATE_PAYOFF_CAP) / CRATE_PAYOFF_CAP

    crate_dist = _bfs_distance(field, occupied, pos, lambda p: _is_crate_adjacent(field, p))
    features[IDX_CRATE_DIST] = (min(crate_dist, CRATE_DIST_CAP) / CRATE_DIST_CAP
                                 if crate_dist is not None else 1.0)

    features[IDX_PAYOFF_X_SAFE] = features[IDX_CRATE_PAYOFF] * features[IDX_SAFE_BOMB]

    # Opponent-aware features (Experiment 24) -- see N_FEATURES docstring.
    opponent_pos = _select_opponent_target(self, pos, others)

    if opponent_pos is not None:
        # An opponent's own tile is in `occupied` (they're standing on it),
        # so it's never reachable as a BFS goal under _is_free's rules --
        # same reason crates use "nearest *adjacent* tile", not the crate
        # tile itself. Target the nearest tile adjacent to the opponent.
        def _adjacent_to_opponent(p):
            return abs(p[0] - opponent_pos[0]) + abs(p[1] - opponent_pos[1]) == 1

        if _adjacent_to_opponent(pos):
            opp_dist, opponent_direction = 0, None
        else:
            opp_dist = _bfs_distance(field, occupied, pos, _adjacent_to_opponent)
            opponent_direction, _ = _bfs_first_step(field, occupied, pos, _adjacent_to_opponent)

        features[IDX_OPPONENT_DIST] = (min(opp_dist, OPPONENT_DIST_CAP) / OPPONENT_DIST_CAP
                                        if opp_dist is not None else 1.0)

        if features[IDX_DANGER] == 0.0:
            for i, action in enumerate(MOVE_ACTIONS):
                if action == opponent_direction:
                    features[IDX_OPPONENT_DIR_BASE + i] = 1.0

        own_blast = get_blast_coords(field, pos, s.BOMB_POWER)
        features[IDX_OPPONENT_IN_BLAST] = 1.0 if opponent_pos in own_blast else 0.0

        if bombs_left:
            occupied_for_opponent = occupied | {pos}  # we block their escape route too
            features[IDX_OPPONENT_TRAPPED] = 0.0 if _can_escape_own_bomb(
                field, occupied_for_opponent, pos, bombs, start_pos=opponent_pos) else 1.0
    # Experiment 27: when there is no opponent, the whole opponent block stays
    # at zero, including IDX_OPPONENT_DIST. It used to be set to 1.0 here on
    # the reading "no opponent == maximally far", which is the intuitive
    # encoding and the wrong one for a linear model.
    #
    # In solo training (Task 1 and Task 2, which is how every submitted
    # checkpoint was trained) `others` is empty on every single step, so that
    # 1.0 was a constant -- and so is IDX_BIAS. Two features that are
    # constantly 1.0 are perfectly collinear: they receive identical gradients
    # on every update, so from a zero initialization they end up with exactly
    # identical weights, and the intended bias is split evenly across the two.
    # Confirmed on the Experiment 26 control checkpoint, where
    # w[:, IDX_BIAS] == w[:, IDX_OPPONENT_DIST] holds exactly, to the bit, for
    # all six actions.
    #
    # That is harmless during solo training and actively wrong at evaluation:
    # with opponents on the board IDX_OPPONENT_DIST is no longer 1.0, so half
    # of what the model learned as a constant bias silently turns into a term
    # that varies with opponent distance and was never fitted as one.
    #
    # Zero is the correct "this feature is inactive" encoding here -- a zero
    # feature contributes exactly 0 to every Q-value and receives exactly zero
    # gradient, so the weight simply stays 0 and the bias stays clean. This is
    # the same class of bug as Experiment 10's "bomb available" feature, which
    # was removed for being collinear with safe-to-bomb; it reappeared in
    # Experiment 24 and went unnoticed until the Experiment 26 control was
    # built.

    return features
