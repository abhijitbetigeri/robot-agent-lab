# Night Shift — the overnight shift in a small biology lab, run by agents

Recursive Agents Hackathon · Crusoe × Cerebral Valley · Sat 2026-10-17 · Shack15
Track: **Shared Brain** (primary), with Done Means Done built in. No physical hardware: MuJoCo sim of the SO-101 arm.

**Use case.** Sample tubes arrive all day with colored caps. Before the night runs, each tube must be in the right place: red caps → cold rack, blue caps → sequencer rack, green caps are controls and stay on the bench. A technician does this twice a night. A tube in the wrong rack ruins a run that costs more than the robot. We replace the technician's night shift with an agent team you can walk away from.

In the sim: blocks are tubes, bin patches are racks, green has no rack on purpose. Each episode is one shift.

One line for judges: *A team of Nemotron agents runs a lab's night shift on a simulated robot arm. Every agent reads a shared, time-versioned lab notebook before its shift and writes back after. A sealed camera verifier decides what "done" means. We show, with numbers, that shift four is cheaper and more honest than shift one.*

Goal the human types, once: **"Put tonight's samples where they belong."**

---

## 0. Hard constraints from the guide

- Submissions due **5:00 PM**. Hacking starts 10:45. Treat **4:40 PM** as our deadline.
- Public repo. 1-minute demo video. 3-minute live demo + Q&A.
- Judging: Technicality 40 · Creativity 25 · Live demo 20 · Future potential 15.
- Anti-projects: no dashboard as the main feature, no Streamlit, no RAG-only. The eval loop and the sim are the demo.
- **New work only.** Disclose on the slide and in README: the supersession/decay idea comes from our earlier Travel-Companion and "Superseded" projects; all code here is written today. The SO-101 model is from MuJoCo Menagerie (Apache-2.0, DeepMind).

## 1. Sponsor map (every one of the four is load-bearing)

| Sponsor | Role in the build | Prize it targets |
|---|---|---|
| **Crusoe Managed Inference** | Every model call. One endpoint, prefix caching measured via `usage.prompt_tokens_details.cached_tokens`. | Main prize |
| **NVIDIA Nemotron** | Super = planner. Lightning = operator + researcher. Nano Omni = sealed vision verifier. Nano = monitor, curator, eval sweeps. | RTX 5090 (Best use of Nemotron) |
| **NVIDIA NemoClaw + OpenShell** | The operator (the only agent that executes generated code) runs inside an OpenShell sandbox whose network policy allows only `api.inference.crusoecloud.com`. Demo a denied egress and the audit log. | Brev credits |
| **You.com** | Researcher agent calls `/v1/search` when the operator hits an unknown (MuJoCo API error, actuator semantics). Result becomes a dated, sourced fact in the brain. | You.com credits ride with placement |

### Model IDs (confirm every one against `GET /v1/models` before 11:00)

| Role | Model ID | Flags |
|---|---|---|
| Planner (writes DoD, re-plans) | `nvidia/Nemotron-3-Super-120B-A12B` | `extra_body={"chat_template_kwargs":{"enable_thinking":true}}`, `max_tokens>=2048`, temp 1.0 / top_p 0.95 |
| Operator (tool calls into sim) | `nvidia/nemotron-3.5-lightning-30b-a3b` | `enable_thinking:false`, `force_nonempty_content:true`, tools |
| Researcher (You.com) | `nvidia/nemotron-3.5-lightning-30b-a3b` | tools |
| Verifier (image → verdict) | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` | image as base64 data URI, `max_tokens>=1024`, 3 samples, majority vote |
| Monitor, Curator, eval sweeps | `nvidia/Nemotron-3-Nano-30B-A3B` | `response_format` JSON if `supported_parameters` lists it |
| Fallback for tool calling if Lightning misbehaves | `openai/gpt-oss-120b` | Crusoe says most reliable tool caller. Only as fallback; log the swap. |

Gotchas from Crusoe's skill file: key may contain `$` → export in single quotes. IDs are case-sensitive. Append the assistant `tool_calls` message before `role:"tool"` messages. Reasoning tokens count against `max_tokens`. Rate limits are per model: 30 RPM without a card, 600 with. **Add a card** (also needed for the free credits). Spread roles across models = more headroom.

## 2. Architecture

```
goal: "Put tonight's samples where they belong."   (red→cold rack, blue→sequencer rack, green stays)
          │
   ┌──────▼──────┐   reads brain      ┌────────────────┐
   │  PLANNER    │◄──────────────────►│  SHARED BRAIN  │  SQLite ledger
   │  (Super)    │  writes DoD        │  facts/procs/  │  bi-temporal triples
   └──────┬──────┘                    │  insights      │  provenance, decay
          │ DoD (JSON checks)         └───▲────────▲───┘
   ┌──────▼──────┐  tool calls             │        │ deltas (ACE curator, Nano)
   │  OPERATOR   │──────────► SIM (MuJoCo) │        │
   │ (Lightning, │◄────────── obs, frames  │   ┌────┴─────┐
   │  in OpenShell)                        │   │ RESEARCHER│ (Lightning + You.com)
   └──────┬──────┘                         │   └──────────┘
          │ claims "done" + trace          │
   ┌──────▼──────┐  sealed: sees ONLY the verifier camera frame + DoD
   │  VERIFIER   │  (Nano Omni) ──► PASS / FAIL per check, with reason
   └──────┬──────┘
   ┌──────▼──────┐  loop/drift/budget → continue | re-plan | stop | ping human
   │  MONITOR    │  (Nano)
   └─────────────┘
   ground truth from sim scores everything, including the verifier itself.
```

### 2.1 Sim (`sim/`)
- Menagerie `robotstudio_so101/scene.xml` + 3 colored boxes (free joints) + 2 bin patches (non-colliding colored geoms). Two cameras: `operator_cam` (arm view) and `verifier_cam` (top-down). Already verified: `mujoco 3.16`, offscreen `Renderer` works on this Mac under `uv run --python 3.12`.
- Tools exposed to the operator (and nothing else):
  - `scene_state()` → joint angles, gripper opening. **No block positions.**
  - `look()` → operator_cam frame (jpeg, ~640px)
  - `move_above(x, y)`, `lower()`, `lift()` → damped-least-squares IK on the gripper site, 6 joints. Fallback: precomputed pose grid.
  - `gripper("open"|"close")`
- Injected gotchas (seeded, per episode): gripper close fails 1 in 3 unless called twice (mirrors the real motor 6); bin B moves +8 cm from episode 3 on; verifier_cam gets a 3 cm offset in episode 5.
- `ground_truth()` → used only by `eval/`, never by agents.

### 2.2 Brain (`brain/`)
- `ledger.py`: SQLite. Row = `(id, subject, relation, object, tier, valid_from, valid_to, source_agent, episode, evidence_ref, confidence, helpful, harmful)`. Tiers: `fact` (world/hardware), `procedure` (how-to), `insight` (lesson).
- Supersession rule: a new row with the same `(subject, relation)` closes the old row's `valid_to`. History kept. Deterministic, no LLM call.
- `retrieve.py`: filter `valid_to IS NULL`, score `relevance × exp(-age_episodes / window) × (1 + helpful − harmful)·ε`. Return top-k as a compact "briefing" the planner and operator get as the *last* message (keeps the cached prefix stable).
- `curator.py` (Nano): after each episode, reads the trace + verifier verdicts and emits **deltas only**: `add | supersede | upvote | downvote`. Never rewrites the brain wholesale.
- `read_before / write_after` hooks are the only way agents touch it.

### 2.3 Done Means Done (`agents/`)
- Planner writes the DoD as JSON: `[{id, check, how_to_verify: "frame"|"state", evidence_required: true}]`. Frozen for the episode; operator cannot edit it.
- Verifier is sealed: gets the DoD + verifier_cam frame only. Never sees the operator's messages. Writes reasoning then verdict; 3 samples, majority.
- Operator's "done" claim must cite tool-call IDs per DoD item. Monitor diffs the claim against the actual tool log.
- Monitor rules: loop = same tool + same observation twice (DeLoop); per turn operator reports `[lo, hi]` completion + `infeasible` flag (BAGEN); re-inject the DoD every 6 turns (plan drift); budget cap in tokens and turns. Decision: `continue | replan | stop | ping`.
- `ping` is rare and real: writes a question to `inbox.md` and halts. Demo it once (camera unplugged).

### 2.4 Eval (`eval/`)
- Fixed seeds, same models, same prompts in both arms (MemDelta). Arms: **no-brain** (fresh agents every episode) vs **brain**.
- 8 episodes per arm for the live table; 30+ per arm with Nano for the README.
- Metrics: tokens per episode, tool calls per episode, episodes to first true success, false "done" claims caught, verifier accuracy vs ground truth, stale-fact servings after bin move, cache hit rate from Crusoe usage fields.
- STALE-style mini-set: 10 scenarios where a fact silently stops being true (bin moved). Report serving rate with and without supersession.

### 2.5 NemoClaw (`nemoclaw/`) — timebox 45 min, start only after 2.3 works
- Needs Node ≥ 22.19 (this Mac has 20: `brew install node@22` or `nvm use 22`), Docker Desktop running.
- `curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash && nemoclaw onboard` → provider "Other OpenAI-compatible", URL `https://api.inference.crusoecloud.com/v1`, model Lightning.
- Policy: allow only `api.inference.crusoecloud.com:443`. The operator runs `tools` inside; attempt a `curl example.com` from it on stage → denied, show audit log.
- If install fights back past 45 min: ship the policy YAML + README section explaining the design, mark as "not demoed", move on.

## 3. Papers → components (what we actually implement, and what we cite)

| Component | Paper | What we take |
|---|---|---|
| Brain write path | **ACE** 2510.04618 | Curator emits itemized deltas; no wholesale rewrite; helpful/harmful counters |
| Supersession | **Temporal Validity in Retrieval Memory** 2606.26511 | Deterministic triple supersession in a bi-temporal ledger; stale-fact serving as the headline metric |
| Fact versioning | **Zep/Graphiti** 2501.13956 | `valid_from / valid_to`, invalidate don't delete |
| Tiers | **G-Memory** 2506.07398 | facts / procedures / insights; role-aware retrieval |
| Provenance | **Collaborative Memory** 2505.18279 | `(source_agent, episode, evidence_ref)` on every row |
| Retrieval score | **Generative Agents** 2304.03442 + our decay thesis | recency × relevance × usefulness |
| Lean | **ExpeL** 2308.10144 | upvote/downvote insights; prune at `harmful > helpful` |
| Sealed verifier | **Self-Authored Verification Is Unreliable** 2607.24300 | verifier never sees worker reasoning |
| Verifier format | **Generative Verifiers** 2408.15240 | reason → verdict, vote 3 |
| Claim vs trace | **OverclaimBench** 2609.20812 | diff "done" claim against tool log |
| Evidence-carrying done | **When May an Agent Stop** 2608.23623 | each DoD item cites a tool-call ID |
| Budget / stop | **BAGEN** 2606.00198 | `[lo,hi]` completion + infeasible flag each turn |
| Loop detection | **DeLoop** 2610.19376 | same tool + same observation = loop |
| Plan drift | **From Plan to Action** 2604.12147 | re-inject DoD every k turns |
| Ask early | **Ask Early, Ask Late, Ask Right** 2605.07937 | clarification only before the DoD is written |
| Fair eval | **MemDelta** 2606.29914 | hold model + prompts fixed across arms |
| Stale test set | **STALE** 2605.06527 | 10 implicit-invalidation scenarios |
| Report format | **DolphinBench** 2609.24971 | accuracy vs tokens, not one number |

Put this table in the README. Cite-only on the slide: ACE, Temporal Validity, G-Memory, OverclaimBench, BAGEN.

## 4. Timeline (hacking 10:45 → submit 4:40)

| Block | Time | Deliverable | Owner |
|---|---|---|---|
| Setup | 10:45–11:15 | Card on Crusoe, credits form, key (expiry Oct 18), `/v1/models` dump saved to `llm/models.json`, one call per role model succeeds, repo public, `uv` env, Menagerie sparse clone, You.com key | all |
| Sim | 11:15–12:15 | `sim/env.py` tools + IK + 2 cameras + gotcha injection + `ground_truth()`; a scripted sort succeeds | A |
| Agents | 11:15–13:00 | `llm/crusoe.py` (backoff, usage logging), planner → operator → verifier → monitor on one episode end-to-end, no brain | B |
| Lunch | 13:00–13:20 | eat at the desk | |
| Brain | 13:20–14:30 | ledger, retrieve, curator, hooks, researcher + You.com, supersession test passes | B |
| Eval | 14:30–15:20 | `eval/run_sweep.py`, 8+8 episodes, metrics table auto-generated to `eval/report.md` | A |
| NemoClaw | 15:20–16:05 | sandboxed operator + denied-egress demo (timeboxed) | A |
| Polish | 15:20–16:05 | README with disclosure + paper table, demo script, `ping` scenario | B |
| Video + submit | 16:05–16:40 | 1-min screen recording, submission form, team members added | all |
| Buffer | 16:40–17:00 | nothing new; rehearse the 3 minutes | |

Solo? Collapse A/B into one lane and drop NemoClaw to a README section, researcher to one call, eval to 5+5 episodes.

## 5. Repo layout

```
robot-agent-lab/
  README.md            pitch, disclosure, paper table, results
  PLAN.md              this file
  llm/crusoe.py        client, model registry, backoff, usage + cache logging
  llm/models.json      dump of /v1/models at 11:00
  sim/scene.xml        menagerie so101 + blocks + bins + 2 cams
  sim/env.py           tools, IK, gotchas, ground_truth
  brain/ledger.py      sqlite, supersession
  brain/retrieve.py    scoring, briefing
  brain/curator.py     ACE deltas
  agents/planner.py operator.py verifier.py monitor.py researcher.py
  run_episode.py       one goal → DoD → loop → verdict → curator
  eval/run_sweep.py    arms, seeds, metrics → eval/report.md
  eval/stale_set.json  10 scenarios
  nemoclaw/policy.yaml README.md
  demo/script.md       3-minute script, 1-minute video shot list
```

## 6. Demo script (3 minutes)

1. (0:00) One line of goal typed. Walk away. Fresh agent, no brain: watch it rediscover the flaky gripper, claim "done" with a red block in the blue bin. Verifier frame on screen: **FAIL**, reason shown.
2. (0:50) Same goal, brain on. Briefing on screen: "gripper: retry close ×2 (ep 1, operator)". First try succeeds. Verifier **PASS**.
3. (1:30) Move bin B live. Brain supersedes `binB.position` after one failure; old row shows `valid_to`. Next agent never looks in the old place. Stale-fact servings: 0.
4. (2:10) Table: no-brain vs brain, tokens, tool calls, false-dones caught, verifier accuracy vs ground truth, cache hit rate.
5. (2:40) Unplug the verifier camera. Monitor: `ping`. One line in `inbox.md`. "That is the only time today it needed a human."

Hand the laptop to a judge for step 3 if the room allows it.

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| IK is flaky | precomputed pose grid over the table (10×10), nearest pose |
| Omni misreads frames | add colored-centroid pre-processing text alongside the image; still sealed; keep Omni as the judge |
| Lightning tool calls degrade | swap operator to `gpt-oss-120b`, log the swap, keep Nemotron everywhere else |
| 429s | per-model spread already; add card; backoff from skill file |
| NemoClaw install stalls | ship policy + README, mark not demoed |
| Running late at 15:30 | cut researcher and NemoClaw; brain + verifier + eval table is the project |

## 8. Setup commands

```bash
# python
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mujoco pillow numpy openai httpx
# so101 model
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie vendor/menagerie
git -C vendor/menagerie sparse-checkout set robotstudio_so101
# keys (single quotes!)
export CRUSOE_API_KEY='...'
export YOU_API_KEY='...'
curl -s https://api.inference.crusoecloud.com/v1/models -H "Authorization: Bearer $CRUSOE_API_KEY" > llm/models.json
# node for nemoclaw (later)
brew install node@22   # or nvm use 22
```
