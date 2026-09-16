# sarsa_threat — Experiment 29, opponent-threat features

Not a separate model. Same architecture as `sarsa_agent`, plus two features
(`IDX_THREAT_HERE`, `IDX_TRAPPED_BY_OPP`) that model a bomb an opponent has
*not dropped yet*. Every other danger computation in the codebase reasons only
about bombs already placed, and the opponents' own `bombs_left` flag had never
been read.

`sarsa_threat_s0/1/2` are the same code at three training seeds, so that
training-side variance is measured rather than assumed — every earlier
experiment in this log trained once, at seed 0, and reported only the spread
over evaluation boards.
