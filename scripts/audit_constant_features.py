"""Find features that are constant during solo training -- the Experiment 27 defect class.

A feature that never varies across a solo training run cannot be learned as a
feature. If it is constantly zero it is merely dead weight: zero contribution,
zero gradient, weight stays at its initial value. If it is constantly *non-zero*
it is worse than dead -- it is collinear with IDX_BIAS, receives gradients
identical to the bias on every update, and so silently absorbs a share of the
bias. That is invisible during solo training, because only the sum of the two
weights is identifiable, and it becomes wrong the moment the feature starts
varying at evaluation time (i.e. as soon as opponents are on the board).

Two instances of this have now been found by accident, four experiments apart:
Experiment 10's "bomb available" and Experiment 27's IDX_OPPONENT_DIST. This
script is the mechanical version of that discovery, so a third one does not
have to wait for somebody to notice a control behaving oddly.

States are sampled with the game's own stone-wall lattice and bombs and
explosions at assorted stages, and with `others` empty, which is the condition
every solo training run sees on every step.

Crate density is drawn uniformly from [0, CRATE_DENSITY] per sampled state
rather than fixed at the scenario value. A round *starts* at 0.75 and clears
from there, so sampling only at 0.75 describes the first few steps of a round
and nothing after them -- and it badly distorts firing rates for any feature
whose value depends on having somewhere to walk, since at 0.75 nearly every
neighbouring tile is a crate. Measured both ways while adding Experiment 29's
features: the per-direction threat dimensions read 0.5-0.7% at fixed 0.75 and
several times that across the range, which is the difference between
"discard this feature" and "keep it".

With --opponents N the same sampler places N bomb-carrying opponents on the
board, which is the condition the curriculum stage of Experiment 28 trains in
and the condition the tournament runs in. That mode also prints how often each
feature actually fires, because a feature that is almost never on cannot be
learned from however correct it is -- the lesson of Experiment 26, where the
escape-redundancy dimension fired in 1.6% of states and duly learned a weight
of nothing.

Run: python scripts/audit_constant_features.py [--agent qlearn_agent]
                                               [--states 3000] [--opponents 3]
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import settings as s  # noqa: E402


class _Stub:
    """Stands in for the persistent `self` the framework hands the callbacks."""

    def __init__(self):
        self.current_target = None
        self.escape_direction = None
        self.opponent_target_name = None


def random_state(rng, n_opponents=0):
    field = np.zeros((s.COLS, s.ROWS), int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if x % 2 == 1 and y % 2 == 1:
                field[x, y] = -1
    free = [(x, y) for x in range(1, s.COLS - 1) for y in range(1, s.ROWS - 1)
            if field[x, y] == 0]
    density = float(rng.uniform(0.0, s.SCENARIOS["classic"]["CRATE_DENSITY"]))
    for (x, y) in free:
        if rng.random() < density:
            field[x, y] = 1

    open_tiles = [t for t in free if field[t] == 0]
    if len(open_tiles) < 6:
        return None

    idx = rng.choice(len(open_tiles), size=min(6, len(open_tiles)), replace=False)
    picks = [open_tiles[i] for i in idx]
    pos = picks[0]

    bombs = [(t, int(rng.integers(0, s.BOMB_TIMER + 1)))
             for t in picks[1:1 + int(rng.integers(0, 3))]]
    explosion_map = np.zeros((s.COLS, s.ROWS))
    for t in picks[4:4 + int(rng.integers(0, 2))]:
        explosion_map[t] = rng.integers(1, s.EXPLOSION_TIMER + 1)

    coins = [t for t in picks[5:] if rng.random() < 0.5]
    bombs_left = bool(rng.integers(0, 2))

    others = []
    if n_opponents:
        spare = [t for t in open_tiles if t != pos]
        if len(spare) >= n_opponents:
            oi = rng.choice(len(spare), size=n_opponents, replace=False)
            others = [("opp%d" % k, 0, bool(rng.integers(0, 2)), spare[j])
                      for k, j in enumerate(oi)]

    return {
        "field": field,
        "bombs": bombs,
        "explosion_map": explosion_map,
        "coins": coins,
        "self": ("me", 0, bombs_left, pos),
        "others": others,
        "step": int(rng.integers(1, 400)),
        "round": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="qlearn_agent")
    ap.add_argument("--states", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponents", type=int, default=0,
                    help="place N opponents on the board (default 0 = the "
                         "solo-training condition this audit is about)")
    args = ap.parse_args()

    mod = __import__("agent_code." + args.agent + ".callbacks",
                     fromlist=["state_to_features", "N_FEATURES"])
    state_to_features = mod.state_to_features
    n_features = mod.N_FEATURES

    names = {}
    for attr in dir(mod):
        if attr.startswith("IDX_"):
            names.setdefault(getattr(mod, attr), attr)

    rng = np.random.default_rng(args.seed)
    rows = []
    stub = _Stub()
    while len(rows) < args.states:
        gs = random_state(rng, args.opponents)
        if gs is None:
            continue
        rows.append(state_to_features(stub, gs))

    F = np.array(rows)
    lo, hi = F.min(axis=0), F.max(axis=0)
    constant = np.isclose(lo, hi)

    print("agent: " + args.agent + "   states: " + str(len(F))
          + "   features: " + str(n_features)
          + "   opponents: " + str(args.opponents))
    print()

    bad = []
    for i in range(n_features):
        if not constant[i]:
            continue
        val = float(lo[i])
        label = names.get(i, "")
        if i == 0:
            kind = "OK    (the bias itself)"
        elif val == 0.0:
            kind = "dead  (constantly 0 -- never learned, but harmless)"
        else:
            kind = "DEFECT (constantly " + str(val) + " -- COLLINEAR WITH BIAS)"
            bad.append((i, label, val))
        print("  idx " + str(i).rjust(2) + "  " + label.ljust(24) + kind)

    if args.opponents:
        print()
        print("  firing rates (fraction of states where the feature is non-zero):")
        for i in range(n_features):
            rate = float((F[:, i] != 0).mean())
            label = names.get(i, "")
            flag = "   <-- too sparse to learn" if 0 < rate < 0.03 else ""
            print("    idx " + str(i).rjust(2) + "  " + label.ljust(24)
                  + format(100 * rate, "6.1f") + "%" + flag)

    varying = int((~constant).sum())
    print()
    print("  " + str(varying) + " of " + str(n_features) + " features vary; "
          + str(int(constant.sum())) + " constant")

    if bad:
        print()
        print("FAIL: " + str(len(bad)) + " feature(s) collinear with the bias in solo training.")
        print("A constantly non-zero feature splits the learned bias with IDX_BIAS and")
        print("turns into an unfitted varying term as soon as it stops being constant.")
        print("Encode 'this feature does not apply' as 0.0, not as a sentinel value.")
        return 1

    print()
    print("PASS: no feature is constantly non-zero in the solo-training condition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
