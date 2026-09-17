# Submission checklist

Everything the project brief asks for, in the order the two deadlines fall.
Page references are to `final_project.pdf`.

---

## Deadline 1 — agent code, Mon 21.09.2026, 21:00

Section 1, "Submission", page 2. The brief asks for exactly one thing to be
uploaded, plus three account settings that are easy to overlook.

### The upload

| | |
|---|---|
| File | `final-project-agent-code.zip` (in this folder) |
| Where | MaMpf |
| Contents required | "A directory containing the agent code of your best performing model, including all trained parameters. This is the subdirectory of `agent_code`..." |

**Status: ready.** The archive contains one top-level directory, `sarsa_agent/`:

| File | Purpose |
|---|---|
| `callbacks.py` | `setup` and `act` — loaded for every game |
| `train.py` | `setup_training`, `game_events_occurred`, `end_of_round` — only loaded with `--train` |
| `model.pt` | the submitted weights (the checkpoint `callbacks.py` loads) |
| `model_exp30_curriculum_s1.pt` | same weights under their experiment name |
| `model_exp28_curriculum_rb3.pt` | earlier candidate, kept so the choice is reversible |
| `model_exp27_decollinearized.pt` | earlier candidate, and the better agent on the two solo tasks |

`requirements.txt` sits at the root of the archive, not inside `sarsa_agent/`,
because the graders install it at step 2 and only look for the agent directory
at step 3. It lists `numpy` unpinned: their environment already has it from
conda, so `pip install -r` reports "already satisfied" and changes nothing,
whereas a pinned version could install a second copy over the conda one.
Verified in the course container -- numpy 2.4.6 before and after, same path.

Verified before packing:

- Unzipped into a pristine copy of the framework and run exactly as section 8
  describes the graders will run it — a single game, `self.train = False`,
  against three `random_agent`s. Exit 0, no errors.
- Ten rounds: exit 0, no errors, **no step exceeded the 0.5 s limit**.
- Also run inside the Docker container from section 8 (see
  `scripts/docker_submission_test.sh`): exit 0, no errors, agent log written,
  and no training callbacks invoked, confirming `self.train = False`.
- No `__pycache__`, `.DS_Store` or other junk in the archive.
- No absolute paths anywhere in the agent code (section 8 calls this out as a
  common error); every path is built from `os.path.dirname(__file__)`.
- `requirements.txt` included and exercised: the full section 8 sequence was
  run in the container, including `pip install -r requirements.txt` at step 2,
  with numpy unchanged afterwards and the game exiting 0.
- No `multiprocessing` anywhere in the agent, per section 1 "Development".

### The account settings — not done, and only you can do them

Same paragraph of section 1, and each is a hard requirement:

- [ ] Set **Anzeigename / display name** in MaMpf to your real name.
- [ ] Set **Name in Uebungsgruppen / name in tutorials** in MaMpf to the same
      name, identical to your name in **muesli**.
- [ ] **Join your team's submission via the invitation code**, before the
      deadline. Instructions: https://mampf.blog/handing-in-homework-assignments
- [ ] Announce your team, with a team name for the tournament, at
      https://tinyurl.com/fml-final-project-teams

---

## Deadline 2 — report, Mon 28.09.2026, 21:00

Section 1 page 2, and section 9 page 11. **Not started.**

- [ ] **PDF report**, about 4000 words per team member and "not much more",
      not counting title page.
- [ ] **Mark each chapter or section with its main author.** The brief says
      this twice (sections 1 and 9) and gives the reason: so each team
      member can be graded individually, "important for legal reasons".
- [ ] **Seven sections, in this order** (section 9): Introduction, Background,
      Project planning, Methods, Training, Experiments and Results,
      Conclusion. Experiments and Results is explicitly "the most important
      section of the report".
- [ ] **Include the URL of the public repository** in the report.
- [ ] **Make the repository public.** It is currently private, so the URL
      would not open for a grader.
- [ ] **Do not upload the report to the repository** (section 1, explicit).
- [ ] **Do not use the university logo** in the report (section 9, explicit).

Material to write from: `docs/experiments.md` opens with a section called
"Writing the report from this log" — a mapping from each required report
section to the experiments that supply it, every headline number in one
table, an index of the six figures in `figures/`, and a drafted Background.
It also lists the two things the log cannot supply, which are Project
planning and the per-section author attribution.

---

## Rebuilding the archive

If the agent changes, rebuild rather than editing the zip:

The file list is explicit rather than a glob. `agent_code/sarsa_agent/` also
holds five older checkpoints from Experiments 24-26 that are kept in the
repository for reproducibility but have no business in the submission -- a
`*.pt` glob would sweep them in.

```bash
python - <<'EOF'
import shutil, os, tempfile
SHIP = ['callbacks.py', 'train.py',
        'model.pt',                            # the weights callbacks.py loads
        'model_exp30_curriculum_s1.pt',        # same weights, experiment name
        'model_exp28_curriculum_rb3.pt',       # earlier candidate
        'model_exp27_decollinearized.pt']      # earlier candidate
src = 'agent_code/sarsa_agent'
tmp = tempfile.mkdtemp()
dst = os.path.join(tmp, 'sarsa_agent')
os.makedirs(dst)
for f in SHIP:
    shutil.copy(os.path.join(src, f), dst)
shutil.copy('requirements.txt', tmp)   # ROOT of the archive, not inside sarsa_agent/
shutil.make_archive('submission/final-project-agent-code', 'zip', tmp)
EOF
```

The archive must come out as exactly this, and nothing else:

```
requirements.txt
sarsa_agent/callbacks.py
sarsa_agent/train.py
sarsa_agent/model.pt
sarsa_agent/model_exp30_curriculum_s1.pt
sarsa_agent/model_exp28_curriculum_rb3.pt
sarsa_agent/model_exp27_decollinearized.pt
```

Then re-run the submission test before uploading:

```bash
bash scripts/docker_submission_test.sh submission/final-project-agent-code.zip
```
