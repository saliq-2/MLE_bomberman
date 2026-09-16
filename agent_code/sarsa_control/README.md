# sarsa_control — the Experiment 26 control condition

Not a submission candidate. This directory exists so Experiment 26 can
attribute its result to the right cause.

Experiment 26 adds two escape-robustness features and retrains from scratch.
Comparing that against `model_task2_baseline.pt` would confound two changes at
once, because the baseline checkpoint was trained under the 23-feature code of
Experiment 18 and only zero-padded forward since. A difference between them
could be the new features, or it could be everything that changed in the code
base in between — most of all Experiment 24's seven opponent-aware features,
which the baseline's weights have never seen.

So this is the missing control: the feature set exactly as it stood *before*
Experiment 26 (30 features), trained fresh under the identical protocol
(`classic`, single agent, 1000 rounds, seed 0). `callbacks.py` and `train.py`
here are byte-for-byte copies of `sarsa_agent`'s files at commit HEAD, verified
by checksum rather than by inspection — deliberately unmodified, including no
added header comment, so that "identical except for the two features" is a
property that can be checked mechanically instead of trusted.

With this in place the comparison is a clean one:

| Condition | Features | Trained |
|---|---|---|
| `sarsa_control` | 30 | fresh, same protocol |
| `sarsa_agent` (Exp 26) | 32 | fresh, same protocol |

Any difference is then attributable to the two added dimensions alone.
