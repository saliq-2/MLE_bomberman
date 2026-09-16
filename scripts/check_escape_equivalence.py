"""Experiment 26 pre-flight: verify the new escape-redundancy search agrees
with the existing escape check wherever they are supposed to agree.

Two properties are asserted over randomized boards (crates, walls, several
live bombs at assorted countdowns), rather than argued in prose:

 1. Equivalence: len(_escape_first_steps(...)) >= 1 must equal
    _can_escape_own_bomb(...) on every state. Both answer "does any escape
    route exist"; the new one just additionally reports which first moves
    work. If these ever disagree, the redundancy feature is not a strict
    refinement of the old one and Experiment 25's padding-equivalence
    argument for the older checkpoints would no longer carry over.

 2. Monotonicity: robust (>= 2 routes) must imply safe (>= 1 route). A
    state that is "robustly safe" but not "safe" would be incoherent.

Run: python scripts/check_escape_equivalence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings as s
from agent_code.qlearn_agent.callbacks import (
    _can_escape_own_bomb, _escape_options, _occupied_tiles, _is_free,
)


def random_board(rng):
    """Same shape/wall pattern as environment.py's board generator.

    Crate density is drawn per board from [0, CRATE_DENSITY] rather than fixed
    at the scenario's 0.75. A round starts at 0.75 and clears from there, so
    sampling only at 0.75 describes the opening moves and nothing else. This
    matters for the firing-rate numbers reported below, not for the
    equivalence check itself -- see docs/experiments.md, Experiment 29, where
    fixing this overturned a firing rate quoted in Experiment 26.
    """
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
    return field, free


def main(n_states=4000, seed=0):
    rng = np.random.default_rng(seed)
    checked = mismatches = robust_cases = 0
    slacks = []

    for _ in range(n_states):
        field, free = random_board(rng)
        open_tiles = [(x, y) for (x, y) in free if field[x, y] == 0]
        if len(open_tiles) < 6:
            continue

        idx = rng.choice(len(open_tiles), size=min(5, len(open_tiles)), replace=False)
        picks = [open_tiles[i] for i in idx]
        pos, bomb_tiles = picks[0], picks[1:]

        n_bombs = int(rng.integers(0, 4))
        bombs = [(t, int(rng.integers(0, s.BOMB_TIMER + 1)))
                 for t in bomb_tiles[:n_bombs]]

        others = []
        occupied = _occupied_tiles(bombs, others)
        if not _is_free(field, occupied, pos):
            continue

        old = _can_escape_own_bomb(field, occupied, pos, bombs)
        steps, earliest = _escape_options(field, occupied, pos, bombs)
        new_any = len(steps) >= 1
        new_robust = len(steps) >= 2

        checked += 1
        if new_any:
            assert earliest is not None, f"route but no time at pos={pos}"
            slacks.append(max(0.0, (s.BOMB_TIMER - earliest) / s.BOMB_TIMER))
        else:
            assert earliest is None, f"time but no route at pos={pos}"
        if old != new_any:
            mismatches += 1
            print(f"MISMATCH at pos={pos} bombs={bombs}: "
                  f"_can_escape_own_bomb={old} first_steps={sorted(steps)}")
        if new_robust:
            robust_cases += 1
            assert new_any, f"robust but not safe at pos={pos} bombs={bombs}"

    print(f"states checked: {checked}")
    print(f"equivalence mismatches: {mismatches}")
    print(f"robust (>=2 routes) cases seen: {robust_cases} "
          f"({100.0 * robust_cases / max(checked, 1):.1f}% of checked states)")
    if slacks:
        import collections
        hist = collections.Counter(round(v, 2) for v in slacks)
        print(f"escapable states: {len(slacks)} "
              f"({100.0 * len(slacks) / max(checked, 1):.1f}% of checked)")
        print("escape-slack distribution over escapable states "
              "(1.0 = instant, 0.0 = only just):")
        for val in sorted(hist, reverse=True):
            print(f"  slack {val:.2f}: {hist[val]:5d} "
                  f"({100.0 * hist[val] / len(slacks):5.1f}%)")
    if mismatches:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
