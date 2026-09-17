# sarsa_vs_rb3 — Experiment 28, tournament-configuration curriculum

Not a separate model. Same code as `sarsa_agent`; this directory exists only so
the curriculum stage can train without overwriting the submission checkpoint.

Stage 2 of a two-stage curriculum: warm-started from `warm_start.pt` (the
Experiment 27 Task-2 checkpoint) and trained against three `rule_based_agent`s,
i.e. the exact tournament configuration. Stage 1 is the solo Task-2 training
that produced the warm start.

Motivation: every opponent-training run in this log before now started from
zero weights, so it had to learn navigation, bomb safety and opponent play at
the same time inside 1000 rounds. It also predates Experiment 27, meaning the
opponent-aware features added in Experiment 24 have never been trained under a
model whose bias was not split with `IDX_OPPONENT_DIST`.
