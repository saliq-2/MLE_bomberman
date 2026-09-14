"""Throwaway sanity check: does --seed actually vary the board layout across
different seed values on coin-heaven? (Not whether it's reproducible across
runs -- environment.py's `self.rng = np.random.default_rng(args.seed)` being
a single persistent generator already tells us re-running the same seed
reproduces the same round-by-round sequence. The open question is whether
seeds 0..4 differ from *each other*, which is what run_experiment.py's
multi-seed loop relies on to get independent-looking board layouts.)
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import settings as s
from argparse import Namespace
from environment import BombeRLeWorld

for seed in range(5):
    args = Namespace(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir="logs",
                      save_stats=False, match_name=f"seedcheck{seed}", seed=seed,
                      silence_errors=True, scenario="coin-heaven")
    agents = [("qlearn_agent", False)]
    world = BombeRLeWorld(args, agents)
    world.new_round()
    coins = sorted(c.get_state() for c in world.coins)
    print(f"seed={seed} n_coins={len(coins)} first 5 coins: {coins[:5]} "
          f"start_pos={[(a.x, a.y) for a in world.agents]}")
