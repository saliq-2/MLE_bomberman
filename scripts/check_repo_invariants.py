"""Check the invariants that have actually been violated in this project.

Not a general test suite -- each check here corresponds to something that went
wrong at least once, and would have been caught earlier if it had been checked
mechanically:

 1. Every agent's code parses and imports.
 2. Every checkpoint's width equals its own agent's N_FEATURES. (Broken twice:
    once by widening every checkpoint to a single global width regardless of
    the agent, once by reverting a feature and leaving the checkpoints wide.
    Both made the agent fail to start.)
 3. Every IDX_* name train.py imports exists in that agent's callbacks, and
    every index is inside the feature vector. (Feature layouts have been
    renumbered repeatedly; a stale index reads the wrong feature silently.)
 4. Feature indices are a complete, non-overlapping cover of 0..N_FEATURES-1.
 5. state_to_features returns the right length, all finite, in [-1, 1], for
    randomly sampled states with and without opponents.
 6. It is deterministic: identical state in, identical vector out.
 7. No feature is constantly non-zero across a solo-training sample -- the
    Experiment 27 collinearity defect, which cost ~40% of the agent's score
    and was invisible in training.
 8. Per-step decision time is inside the project's 0.5 s budget.

Exit code is non-zero if any check fails, so this can gate a submission.

Run: python scripts/check_repo_invariants.py [--agent NAME ...]
"""
import argparse
import ast
import glob
import importlib
import os
import pickle
import re
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import settings as s  # noqa: E402

# Agents shipped by the framework that do not learn and have no feature vector.
FRAMEWORK_AGENTS = {"random_agent", "peaceful_agent", "rule_based_agent",
                    "coin_collector_agent", "user_agent", "tpl_agent",
                    "fail_agent"}

TIME_BUDGET_S = 0.5

# Agents that are *supposed* to violate an invariant, because reproducing a
# defect is the whole point of them. Exempting by name and reason, rather than
# weakening the check for everybody.
KNOWN_EXCEPTIONS = {
    ("sarsa_control", "solo_collinearity"):
        "Experiment 26/27 control: deliberately keeps the IDX_OPPONENT_DIST = 1.0 "
        "encoding so the defect's cost can be measured against sarsa_control_fixed",
}


class Stub:
    def __init__(self):
        self.current_target = None
        self.escape_direction = None
        self.opponent_target_name = None


def sample_state(rng, n_opponents):
    field = np.zeros((s.COLS, s.ROWS), int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if x % 2 == 1 and y % 2 == 1:
                field[x, y] = -1
    free = [(x, y) for x in range(1, s.COLS - 1) for y in range(1, s.ROWS - 1)
            if field[x, y] == 0]
    density = float(rng.uniform(0.0, s.SCENARIOS["classic"]["CRATE_DENSITY"]))
    for t in free:
        if rng.random() < density:
            field[t] = 1
    open_tiles = [t for t in free if field[t] == 0]
    if len(open_tiles) < 2 + n_opponents + 3:
        return None
    idx = rng.choice(len(open_tiles), size=2 + n_opponents + 3, replace=False)
    p = [open_tiles[i] for i in idx]
    bombs = [(t, int(rng.integers(0, s.BOMB_TIMER + 1)))
             for t in p[1:1 + int(rng.integers(0, 3))]]
    em = np.zeros((s.COLS, s.ROWS))
    for t in p[-1:]:
        em[t] = int(rng.integers(0, s.EXPLOSION_TIMER + 1))
    others = [("o%d" % k, int(rng.integers(0, 5)), bool(rng.integers(0, 2)), p[2 + k])
              for k in range(n_opponents)]
    return {
        "field": field, "bombs": bombs, "explosion_map": em,
        "coins": [t for t in p[-2:-1]],
        "self": ("me", 0, bool(rng.integers(0, 2)), p[0]),
        "others": others, "step": int(rng.integers(1, 400)), "round": 1,
    }


def learning_agents():
    out = []
    for d in sorted(glob.glob(os.path.join(REPO_ROOT, "agent_code", "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name in FRAMEWORK_AGENTS:
            continue
        if not os.path.exists(os.path.join(d, "callbacks.py")):
            continue
        out.append(name)
    return out


def check_agent(name, n_states, rng_seed, failures):
    d = os.path.join(REPO_ROOT, "agent_code", name)

    for f in ("callbacks.py", "train.py"):
        path = os.path.join(d, f)
        if os.path.exists(path):
            try:
                ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError as exc:
                failures.append(f"{name}: {f} does not parse ({exc})")
                return

    try:
        mod = importlib.import_module("agent_code.%s.callbacks" % name)
    except Exception as exc:
        failures.append("%s: callbacks import failed (%s: %s)"
                        % (name, type(exc).__name__, exc))
        return

    n = mod.N_FEATURES
    idx_names = {a: getattr(mod, a) for a in dir(mod) if a.startswith("IDX_")}
    # value -> name, for diagnostics; _BASE entries name their whole span
    by_index = {}
    for a, v in idx_names.items():
        width = len(mod.MOVE_ACTIONS) if a.endswith("_BASE") else 1
        for k in range(v, v + width):
            by_index.setdefault(k, a)

    # 2. checkpoint widths
    for ck in sorted(glob.glob(os.path.join(d, "model*.pt"))):
        try:
            w = pickle.load(open(ck, "rb"))
        except Exception as exc:
            failures.append("%s: %s unreadable (%s)"
                            % (name, os.path.basename(ck), type(exc).__name__))
            continue
        if w.shape != (len(mod.ACTIONS), n):
            failures.append("%s: %s has shape %s, expected %s"
                            % (name, os.path.basename(ck), w.shape,
                               (len(mod.ACTIONS), n)))

    # 3. train.py's imported IDX_* names exist, and all indices are in range
    tp = os.path.join(d, "train.py")
    if os.path.exists(tp):
        src = open(tp, encoding="utf-8").read()
        m = re.search(r"from \.callbacks import \(([^)]*)\)", src, re.S)
        if m:
            for tok in re.findall(r"IDX_[A-Z_]+", m.group(1)):
                if tok not in idx_names:
                    failures.append("%s: train.py imports %s, absent from callbacks"
                                    % (name, tok))
    for a, v in idx_names.items():
        if not (0 <= v < n):
            failures.append("%s: %s = %d outside 0..%d" % (name, a, v, n - 1))

    # 4. indices cover the vector exactly once (BASE names span MOVE_ACTIONS)
    span = {}
    for a, v in idx_names.items():
        width = len(mod.MOVE_ACTIONS) if a.endswith("_BASE") else 1
        for k in range(v, v + width):
            if k in span:
                failures.append("%s: index %d claimed by both %s and %s"
                                % (name, k, span[k], a))
            span[k] = a
    missing = sorted(set(range(n)) - set(span))
    if missing:
        failures.append("%s: indices %s have no IDX_ name" % (name, missing))

    # 5/6. feature vector sanity + determinism
    rng = np.random.default_rng(rng_seed)
    stub = Stub()
    checked = 0
    t0 = time.perf_counter()
    for _ in range(n_states):
        for n_opp in (0, 3):
            gs = sample_state(rng, n_opp)
            if gs is None:
                continue
            f1 = mod.state_to_features(stub, dict(gs))
            if f1 is None:
                failures.append("%s: state_to_features returned None" % name)
                return
            if len(f1) != n:
                failures.append("%s: feature vector length %d, expected %d"
                                % (name, len(f1), n))
                return
            if not np.all(np.isfinite(f1)):
                failures.append("%s: non-finite feature value" % name)
                return
            if np.any(f1 < -1.0 - 1e-9) or np.any(f1 > 1.0 + 1e-9):
                bad = [(i, float(v)) for i, v in enumerate(f1)
                       if v < -1 - 1e-9 or v > 1 + 1e-9]
                failures.append("%s: feature outside [-1,1]: %s"
                                % (name, [(by_index.get(i, i), v) for i, v in bad[:4]]))
                return
            f2 = mod.state_to_features(Stub(), dict(gs))
            if not np.array_equal(f1, f2):
                failures.append("%s: state_to_features is not deterministic" % name)
                return
            checked += 1
    elapsed = time.perf_counter() - t0

    # 7. no constantly non-zero feature in the solo condition
    rng = np.random.default_rng(rng_seed + 1)
    rows = []
    while len(rows) < n_states:
        gs = sample_state(rng, 0)
        if gs is not None:
            rows.append(mod.state_to_features(Stub(), gs))
    F = np.array(rows)
    const = np.isclose(F.min(axis=0), F.max(axis=0))
    exempt = KNOWN_EXCEPTIONS.get((name, "solo_collinearity"))
    for i in range(n):
        if i != mod.IDX_BIAS and const[i] and F[0, i] != 0.0:
            msg = ("%s: feature %d (%s) constantly %.3f in solo training "
                   "-- collinear with the bias (Experiment 27)"
                   % (name, i, by_index.get(i, "?"), F[0, i]))
            if exempt:
                print("  %-22s EXPECTED: %s" % ("", msg.split(": ", 1)[1]))
                print("  %-22s reason:   %s" % ("", exempt))
            else:
                failures.append(msg)

    # 8. timing
    per_call_ms = 1000 * elapsed / max(checked, 1)
    if per_call_ms > 1000 * TIME_BUDGET_S * 0.5:
        failures.append("%s: %.1f ms/call is over half the %.0f ms budget"
                        % (name, per_call_ms, 1000 * TIME_BUDGET_S))

    print("  %-22s N=%-3d checkpoints=%-2d  %.3f ms/call   OK"
          % (name, n, len(glob.glob(os.path.join(d, "model*.pt"))), per_call_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append")
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agents = args.agent or learning_agents()
    print("checking %d learning agent(s)\n" % len(agents))
    failures = []
    for a in agents:
        try:
            check_agent(a, args.states, args.seed, failures)
        except Exception as exc:
            failures.append("%s: check crashed (%s: %s)" % (a, type(exc).__name__, exc))

    print()
    if failures:
        print("FAIL -- %d problem(s):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("PASS -- all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
