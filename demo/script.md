# Demo script (3 minutes) and 1-minute video shot list

## Setup before judges arrive
- Terminal 1: repo root, `.venv` active, `CRUSOE_API_KEY` in `.env`.
- Terminal 2: `watch -n 2 'sqlite3 runs/brain/brain.sqlite "select id,subject,relation,object,valid_from,valid_to from facts order by id desc limit 12"'`
- Browser tab: `eval/report.md` rendered, and `runs/brain/frames/` open in Finder for the latest verifier frame.
- Pre-run: `uv run python -m eval.run_sweep --episodes 6` finished, so the table exists even if the live run misbehaves.

## Live (3:00)
0:00  "A lab's night shift. One line typed, then we walk away."
      `uv run python run_episode.py --episode 1 --no-brain --seed 7`
      Narrate the tool log as it scrolls: look → move → lower → close (holding: null) → the operator either retries or
      claims done. Show the verifier frame and its verdict. If it claimed done with a tube misplaced: "That is the false
      'done' the verifier exists for."
0:50  "Same goal, same seed, with the shared brain. This team has worked three shifts."
      `uv run python run_episode.py --episode 4 --seed 7`   (brain at runs/brain.sqlite from the sweep)
      Point at the briefing printed first: the sticky-gripper procedure and the rack position with their shift numbers.
      Point at Terminal 2 when the curator writes.
1:30  "Now the rack moves." Episode 3+ already has the blue rack shifted. Show the ledger: the old `bin_blue position`
      row has `valid_to = 3`, the new row is current. "Nothing deleted. The old fact just lost. Zero stale servings after."
2:10  The table. Read three numbers: tokens per shift, false dones, verifier accuracy vs ground truth.
      "The verifier has a measured error rate because the sim knows where every tube is. Nobody else can tell you that."
2:40  `uv run python run_episode.py --episode 9 --camera-down` → monitor pings. `cat runs/inbox.md`.
      "That is the only time tonight it needed a human."

## Q&A ammo
- Why not a VLM operator? The operator is text; perception is a tool. The *verifier* is the vision model and it is sealed.
- What is "new work"? Everything here, today. The supersession idea is ours from two earlier projects; disclosed in README.
- What breaks? Say it: Lightning sometimes narrates tool calls instead of emitting them; we retry with a bigger budget and log it.
- Why Nemotron? Five roles, five price points; Nano runs the sweep for cents; Omni is the only one that looks.

## 1-minute video (record with QuickTime, no voice needed, captions)
1. (0-10s) Title card: "Robot Agent Lab — one line, then walk away." Goal typed.
2. (10-30s) No-brain shift: sticky gripper, false done, verifier FAIL with reason.
3. (30-45s) Brain shift: briefing on screen, first-try success, verifier PASS.
4. (45-55s) Ledger: rack row superseded, valid_to set.
5. (55-60s) The table. End card: repo URL.
