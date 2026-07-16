# Git workflow

Two people, one repo. This is the whole thing. Nothing here is advanced.

## The rules

1. **Never push to `main`.** Always a branch, always a pull request.
2. **`git pull` before you start anything.** Most beginner pain is skipping this.
3. **Small commits.** One idea each.
4. **Never `git push --force`.** It can delete the other person's work with no warning.
5. **Flag changes to shared files** (`types.py`, `config.yaml`, `main.py`) before merging.
6. **When lost: `git status`.** It tells you where you are and usually what to do next.

## Who owns what

Git only conflicts when two people edit **the same lines of the same file**.
We own different files, so conflicts should be rare:

| Track A (trigger/brain/overlay) | Track B (transcribe/correction/summarize) | Shared — flag before changing |
| --- | --- | --- |
| `trigger.py` | `transcribe.py` | `types.py` |
| `brain.py` | `correction.py` | `config.yaml` |
| `overlay.py` | `summarize.py` | `main.py` |

`audio.py` is Phase 0 — paired, on the Windows machine.

## The loop

Starting a piece of work:

```bash
git checkout main
git pull                        # EVERY time you start. Not optional.
git checkout -b feat/trigger    # your branch, named for what you're doing
```

While working:

```bash
git add meeting_genie/trigger.py
git commit -m "score sentences against config weights"
git push -u origin feat/trigger   # -u only on the FIRST push of this branch
```

After that first push, just:

```bash
git push
```

When it's ready:

1. GitHub shows a **"Compare & pull request"** button. Click it.
2. Write a line about what changed.
3. Request the other person as reviewer.
4. They approve → **Merge pull request** → **Delete branch**.

Then start the next thing:

```bash
git checkout main
git pull                        # picks up what you just merged
git checkout -b feat/brain
```

## Branch naming

- `feat/trigger`, `feat/brain`, `feat/overlay` — Track A
- `feat/transcribe`, `feat/correction`, `feat/summarize` — Track B
- `feat/audio` — Phase 0, shared
- `fix/headset-mic-threshold` — bug fixes

## Commit messages

Say what changed, not "update" or "changes":

```
good:  fix silence threshold to calibrate from noise floor
good:  port Ollama call from prototype
good:  add sentence-splitting before trigger scoring
bad:   update
bad:   stuff
bad:   asdf
```

When you copy code from the old prototype (`Meesha16/Voice_Assistant`), say so
in the message. Keeps the origin traceable.

## The two things that go wrong

### 1. Your branch is behind main

Someone merged while you were working. Catch up:

```bash
git checkout feat/trigger
git fetch origin
git merge origin/main
```

Fix conflicts if any (below), commit, push. Not a big deal.

### 2. Merge conflict

Git puts both versions in the file:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```

Delete the markers. Keep what should survive — often **both**, edited together.
Then:

```bash
git add path/to/file.py
git commit
```

Looks scarier than it is. The file is just text with markers in it.

## Useful commands

```bash
git status                  # where am I, what have I changed
git log --oneline -10       # last 10 commits
git branch                  # which branches exist, * marks current
git diff                    # what changed but isn't staged
git diff --staged           # what's staged and about to be committed
git checkout -- file.py     # THROW AWAY changes to a file (careful)
git stash                   # park uncommitted changes temporarily
git stash pop               # bring them back
```

## Branch protection (optional)

Once the repo has commits, one of us can add it:
**Settings → Branches → Add rule → `main` → require pull request before merging.**
(Newer GitHub UI: **Settings → Rules → Rulesets**.)

Free orgs only get this on **public** repos. It stops accidental pushes to main.
Nice to have, not required — the rules above matter more than the enforcement.
