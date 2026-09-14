"""Experiment 25, part 1: classify each of our self-kills into
(a) an opponent bomb was placed after our own bomb and before/at our death
    (i.e. during our own bomb's ticking+blast window), or
(b) no opponent bomb appeared during that window.

Matching a self-kill to "our own bomb" relies on the game's bombs_left
mechanic: an agent can have at most one live (undetonated) bomb at a time,
so the most recent explosion logged for owner X is unambiguously caused by
X's most recent drop. Verified against environment.py's exact log line
formats (drop/explode/self-kill), not assumed.
"""
import re
import sys
import os
from collections import defaultdict

STEP_RE = re.compile(r"STARTING STEP (\d+)")
ROUND_START_RE = re.compile(r"STARTING ROUND #(\d+)")
DROP_RE = re.compile(r"Agent <([^>]+)> drops bomb at")
EXPLODE_RE = re.compile(r"Agent <([^>]+)>'s bomb at .* explodes")
SELF_KILL_RE = re.compile(r"Agent <([^>]+)> blown up by own bomb")


def classify_log(log_path, our_name, opponent_names):
    results = []
    cur_step = 0
    pending_drop = {}
    last_explosion_drop = {}
    opp_drops = []
    with open(log_path) as f:
        for line in f:
            if ROUND_START_RE.search(line):
                cur_step = 0
                pending_drop = {}
                last_explosion_drop = {}
                opp_drops = []
                continue
            m = STEP_RE.search(line)
            if m:
                cur_step = int(m.group(1))
                continue
            m = DROP_RE.search(line)
            if m:
                owner = m.group(1)
                pending_drop[owner] = cur_step
                if owner in opponent_names:
                    opp_drops.append((owner, cur_step))
                continue
            m = EXPLODE_RE.search(line)
            if m:
                owner = m.group(1)
                if owner in pending_drop:
                    last_explosion_drop[owner] = pending_drop[owner]
                continue
            m = SELF_KILL_RE.search(line)
            if m:
                owner = m.group(1)
                if owner == our_name:
                    t_death = cur_step
                    t_drop = last_explosion_drop.get(owner)
                    if t_drop is None:
                        results.append({"t_drop": None, "t_death": t_death,
                                         "bucket": "unmatched", "n_opp_bombs": None})
                        continue
                    opp_in_window = [(o, st) for (o, st) in opp_drops if t_drop < st <= t_death]
                    bucket = "a" if opp_in_window else "b"
                    results.append({"t_drop": t_drop, "t_death": t_death,
                                     "bucket": bucket, "n_opp_bombs": len(opp_in_window)})
                continue
    return results


if __name__ == "__main__":
    log_path = sys.argv[1]
    our_name = sys.argv[2]
    opponent_names = sys.argv[3:]
    for r in classify_log(log_path, our_name, opponent_names):
        print(r)
