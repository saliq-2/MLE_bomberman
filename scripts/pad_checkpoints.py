"""Zero-pad older checkpoints to the current feature count.

Experiment 25 established the argument, for the 23 -> 30 case, that this is
behaviour-preserving rather than merely convenient: Q(s,a) = w[a] . f(s),
so a weight of exactly 0 on a new index contributes exactly 0 to every
Q-value no matter what that feature reads, provided the newer features were
*appended* without disturbing how the older ones are computed. Experiment 26
appends indices 30-31 (escape robustness) the same way, leaving
_can_escape_own_bomb and every earlier feature untouched -- so the same
argument carries, and the padded checkpoint plays the identical policy to
the one that was trained.

Carried forward here as a reusable script instead of being redone by hand,
and the equivalence is re-verified numerically on every call rather than
taken on faith: for random feature vectors, padded @ f must equal
original @ f[:k].

One wrinkle worth recording, because it cost a confusing failed run: that
identity holds exactly in real arithmetic but *not* bit-exactly in floating
point. numpy blocks a 32-wide dot product differently from a 23-wide one,
so the partial sums are accumulated in a different order and the two
results can differ in the last ulp even though every added term is exactly
0.0. Padding 30 -> 32 happened to come out bit-identical; 23 -> 32 did not.
The deviation is therefore checked against a tight tolerance and its
observed maximum is printed, rather than being asserted to be zero -- the
honest claim is "equal to within floating-point noise, ~1e-16 on Q-values
of order 1", not "bit-identical". The only way that noise could change
behaviour is by flipping an exact tie between two actions' Q-values in the
argmax, which requires the two to be equal to the last bit.

Each agent directory is padded to *its own* `N_FEATURES`, read from that
agent's `callbacks.py`, not to one global width. This matters as soon as the
repo holds agents at different feature counts: Experiment 27's control
conditions (`sarsa_control`, `sarsa_control_fixed`) deliberately run the
30-feature code, and widening their checkpoints to 32 would hand a 32-feature
weight vector to a `state_to_features` that returns 30 numbers -- breaking the
very conditions the experiment's attribution depends on. An earlier version of
this script globbed every checkpoint against a single width and would have done
exactly that.

Narrowing is supported on the same terms, and refused unless the dropped
columns are all zero -- see `pad`.

Run: python scripts/pad_checkpoints.py [--width N] [--dry-run]
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def pad(weights, width):
    """Resize a weight matrix to `width` columns, preserving the policy.

    Widening is the zero-padding Experiment 25 established. Narrowing is the
    same argument read backwards and is allowed *only* when every column being
    dropped is exactly zero -- a zero column contributes exactly 0 to every
    Q-value, so removing it changes nothing, while dropping a non-zero column
    silently changes the policy. Needed because feature counts can go down as
    well as up: Experiment 29 added two features, failed, and was reverted,
    which left checkpoints two columns wider than the code that loads them.
    A mismatch there is not a silent degradation -- it raises at load time and
    the agent fails to start -- but it is exactly the kind of breakage worth
    a guarded tool rather than a manual reshape.
    """
    n_actions, k = weights.shape
    if k == width:
        return weights, False
    if k > width:
        dropped = weights[:, width:]
        if np.any(dropped != 0):
            raise ValueError(
                f"refusing to narrow {k} -> {width}: columns {width}..{k-1} "
                f"are not all zero (max |w| = {np.abs(dropped).max():.6g}), "
                f"so dropping them would change the policy")
        return weights[:, :width].copy(), True
    out = np.zeros((n_actions, width), dtype=weights.dtype)
    out[:, :k] = weights
    return out, True


TOLERANCE = 1e-12


def verify(original, padded, trials=256, seed=0):
    """padded @ f vs original @ f[:k] for arbitrary f.

    Returns the largest absolute discrepancy seen over `trials` random
    feature vectors. See the module docstring for why this is a tolerance
    check and not an equality check.
    """
    rng = np.random.default_rng(seed)
    k = original.shape[1]
    width = padded.shape[1]
    worst = 0.0
    for _ in range(trials):
        f = rng.normal(size=width)
        delta = np.max(np.abs((padded @ f) - (original @ f[:k])))
        worst = max(worst, float(delta))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=None,
                   help="force one target width for every agent "
                        "(default: each agent's own N_FEATURES)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(REPO_ROOT, "agent_code", "*", "model*.pt")))
    changed = skipped = 0
    widths = {}

    for path in paths:
        rel = os.path.relpath(path, REPO_ROOT)
        agent = os.path.basename(os.path.dirname(path))

        width = args.width
        if width is None:
            if agent not in widths:
                try:
                    mod = __import__("agent_code." + agent + ".callbacks",
                                     fromlist=["N_FEATURES"])
                    widths[agent] = mod.N_FEATURES
                except Exception as exc:
                    print("  SKIP " + rel + ": cannot read N_FEATURES ("
                          + type(exc).__name__ + ")")
                    widths[agent] = None
            width = widths[agent]
            if width is None:
                skipped += 1
                continue

        with open(path, "rb") as fh:
            weights = pickle.load(fh)

        try:
            padded, did = pad(weights, width)
        except ValueError as exc:
            print(f"  SKIP {rel}: {exc}")
            skipped += 1
            continue

        if not did:
            print(f"  ok   {rel}: already {width}-wide")
            skipped += 1
            continue

        worst = verify(weights, padded) if padded.shape[1] >= weights.shape[1]             else verify(padded, weights)
        if worst > TOLERANCE:
            print(f"  FAIL {rel}: max |dQ| = {worst:.3e} exceeds {TOLERANCE:.0e}")
            return 1

        verb = "pad " if padded.shape[1] > weights.shape[1] else "trim"
        print(f"  {verb} {rel}: {weights.shape} -> {padded.shape} "
              f"(equivalent, max |dQ| = {worst:.3e})")
        changed += 1
        if not args.dry_run:
            with open(path, "wb") as fh:
                pickle.dump(padded, fh)

    verb = "would change" if args.dry_run else "changed"
    target = args.width if args.width is not None else "per-agent N_FEATURES"
    print(f"{verb} {changed}, left {skipped} alone, target width {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
