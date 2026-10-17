# Robot loop — build spec

Use case: the lab night shift (see PLAN.md). Blocks = sample tubes by cap color, bin patches = racks (`bin_red` = cold rack, `bin_blue` = sequencer rack), green = control, stays on the bench. Each episode = one shift. Names in code stay `block_*` / `bin_*`; the story lives in prompts and README.

The sim is the "bench." Everything the agents can do to it goes through five tools. Everything the eval knows about it comes from `ground_truth()`, which agents never see. Budget: 60 minutes, scripted sort passing before any LLM touches it.

Facts pulled from the Menagerie model (`vendor/menagerie/robotstudio_so101/so101.xml`):
- 6 position actuators, names = joint names: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. Each has `ctrlrange` = joint range. Position servos, kp≈998, so `data.ctrl[i] = target_rad` and step until settled.
- Gripper joint range `-0.17 … 1.75` rad. Treat `1.4` as open, `-0.10` as closed (tune once with a box in the jaws).
- End-effector site: `gripperframe` (inside the `gripper` body). Use it for IK and for "is the block in the jaws."
- Menagerie already added primitive collision geoms and solver params "that work well for manipulation," so physical grasps of a 2 cm cube should hold. `scene_box.xml` is their own grasp test: 2 cm box, `friction="1 .03 .003"`, `condim=3`.
- Reach: shoulder is at z=0.062, upper arm 0.113, lower arm 0.135, wrist+gripper ≈ 0.16. Usable table radius ≈ 0.15–0.30 m in front of the base. Verify with the reach probe below before placing anything.

## 1. Scene (`sim/scene.xml`)

Start from Menagerie `scene.xml` (includes `so101.xml`, floor, light). Add:

```xml
<worldbody>
  <!-- tubes: standing cylinders 2 cm wide, 3 cm tall, free joints -->
  <body name="block_red"   pos="0.22  0.06 0.01"><freejoint/><geom name="g_red"   type="box" size="0.01 0.01 0.01" rgba="0.9 0.1 0.1 1" mass="0.02" condim="3" friction="1 .03 .003"/></body>
  <body name="block_green" pos="0.26 -0.02 0.01"><freejoint/><geom name="g_green" type="box" size="0.01 0.01 0.01" rgba="0.1 0.8 0.1 1" mass="0.02" condim="3" friction="1 .03 .003"/></body>
  <body name="block_blue"  pos="0.19 -0.08 0.01"><freejoint/><geom name="g_blue"  type="box" size="0.01 0.01 0.01" rgba="0.1 0.2 0.9 1" mass="0.02" condim="3" friction="1 .03 .003"/></body>

  <!-- bins: flat colored patches, no collision -->
  <geom name="bin_red"  type="box" size="0.04 0.04 0.001" pos="0.20  0.17 0.001" rgba="0.9 0.3 0.3 0.6" contype="0" conaffinity="0"/>
  <geom name="bin_blue" type="box" size="0.04 0.04 0.001" pos="0.20 -0.17 0.001" rgba="0.3 0.3 0.9 0.6" contype="0" conaffinity="0"/>

  <!-- cameras -->
  <camera name="operator_cam" pos="0.55 0.35 0.35" xyaxes="-0.55 0.84 0  -0.45 -0.3 0.84"/>
  <camera name="verifier_cam" pos="0.22 0 0.70" xyaxes="0 -1 0  1 0 0"/>   <!-- top-down -->
</worldbody>
<keyframe>
  <key name="home" qpos="0 -1.57 1.57 1.2 0 1.4  0.22 0.06 0.01 1 0 0 0  0.26 -0.02 0.01 1 0 0 0  0.19 -0.08 0.01 1 0 0 0"/>
</keyframe>
```

Green has no bin on purpose: the correct plan leaves it on the table. That is the planner's first chance to write a non-obvious definition of done, and the verifier's first chance to catch an operator that "helpfully" drops green somewhere.

Reach probe (run once, 2 minutes): sample 5,000 random joint configs within ranges, forward-kinematics the `gripperframe` site, keep those with the gripper z-axis pointing down (`site_xmat[:,2]·(0,0,-1) > 0.9`) and z in 0.005–0.03. Plot or print the xy hull. Move blocks and bins inside it. Keep the samples: they are the IK fallback (§3).

## 2. Env API (`sim/env.py`)

```python
class LabEnv:
    def __init__(self, seed: int, gotchas: dict): ...
    def reset(self) -> dict                      # keyframe home, randomize block xy ±2 cm, apply gotchas
    # ---- the five operator tools ----
    def scene_state(self) -> dict                # {"joints_deg": {...}, "gripper": "open|closed|unknown", "t": float}
    def look(self) -> str                        # jpeg path of operator_cam, 640x480
    def move_above(self, x: float, y: float) -> dict   # hover at z=0.06 over (x,y); {"ok", "reached_xy_err_m", "steps"}
    def lower(self) -> dict                      # descend to z=0.012 at current xy; {"ok", "contact": bool}
    def lift(self) -> dict                       # back to z=0.06
    def gripper(self, cmd: "open"|"close") -> dict     # {"ok", "holding": bool|None}   (None when gotcha hides it)
    # ---- eval only, never exposed to agents ----
    def ground_truth(self) -> dict               # block xy, which bin each is in, holding, gripper state
    def verifier_frame(self) -> str              # jpeg of verifier_cam, 512x512
```

Rules:
- `scene_state` never returns block positions. The operator has to `look()`.
- Every tool call returns a `call_id` (monotonic int) that the DoD evidence refers to.
- Every tool call appends `{call_id, tool, args, result, t}` to `env.trace`. The monitor reads the trace; the verifier never does.
- `holding` is computed from contacts between either jaw geom and a block geom, both jaws touching. That is also the grasp-success criterion in `ground_truth()`.

## 3. Motion (`sim/motion.py`)

Primary: damped-least-squares IK on the `gripperframe` site, 5 arm joints (`wrist_roll` held at 0). 6-row error: 3 position + 3 orientation (target orientation = gripper z-axis down). 

```python
def ik_step(model, data, target_pos, target_down=True, damping=0.05):
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    err_p = target_pos - data.site_xpos[site_id]
    # orientation: rotate current z-axis onto -world-z
    z_cur = data.site_xmat[site_id].reshape(3,3)[:, 2]
    err_r = np.cross(z_cur, np.array([0,0,-1.0]))        # small-angle
    J = np.vstack([jacp, 0.5*jacr])[:, arm_dofs]           # 6 x 5
    err = np.concatenate([err_p, 0.5*err_r])
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
    return dq
```

Loop: 30 IK iterations on a *copy* of `qpos`, clip to joint ranges, write the result into `data.ctrl[arm]`, then step the sim until `|qpos - ctrl| < 0.02 rad` or 600 steps (~1.2 s at the model's 2 ms timestep). Report `reached_xy_err_m` honestly; the operator sees it.

Fallback (if IK fights you past 20 minutes): the reach-probe samples from §1 as a pose table. `move_above(x,y)` = nearest sample at z=0.06 with gripper down, `lower()` = nearest at z=0.012 with same xy. 10×10 cm bins over the workspace give ~1 cm accuracy, good enough for 2 cm cubes.

Grasp: `gripper("close")` sets `ctrl[gripper] = -0.10`, steps 300, then checks contacts. If physical grasping is unreliable after one tuning pass (try `friction="1.5 .05 .005"` on the cubes, and closing to a value that leaves ~1 mm squeeze), switch to a **weld fallback**: when closed and a block is within 1.5 cm of `gripperframe`, activate an `equality/weld` between gripper and that block; deactivate on open. Say so in the README ("simplified grasp"). Judges care about the agents, not contact physics.

## 4. Gotchas (`sim/gotchas.py`)

Applied in `reset()` from the seed and episode number. These are what the brain must learn.

| Name | Effect | Active | Fact the brain should end up with |
|---|---|---|---|
| `flaky_gripper` | first `close` per grasp returns `holding=None` and does not close; second call works | all episodes | `gripper, needs_retry_on_close, true` |
| `bin_shift` | `bin_blue` moves +0.08 m in y | episodes ≥ 3 | `bin_blue, position, (0.20,-0.09)` supersedes the old row |
| `cam_offset` | `operator_cam` shifted 3 cm; `look()` frames mislead by one block width | episodes ≥ 5 | `operator_cam, xy_bias_m, (0.03,0)` |
| `green_trap` | always on | always | `block_green, has_bin, false` (an insight: "leave green") |

The verifier camera is never perturbed except for the on-stage "unplug" moment (`verifier_frame()` raises; monitor must `ping`).

## 5. Episode loop (`run_episode.py`)

```
reset(seed, episode)
briefing = brain.read(for_role="planner") if brain else ""
dod = planner(goal, briefing)                  # frozen JSON, saved to trace
for turn in range(MAX_TURNS=25):
    msg = operator(dod, briefing, tool_log_tail)   # returns tool_calls OR a done-claim
    if tool_calls: results = env.call(...); continue
    if done_claim:                              # {"done": true, "evidence": {dod_id: [call_ids]}}
        claim_ok = monitor.claim_vs_trace(done_claim, env.trace)
        verdicts = verifier(dod, env.verifier_frame())     # sealed; 3 samples, vote
        break
    decision = monitor(turn, env.trace, operator_report)   # continue | replan | stop | ping
    if decision == "replan": dod = planner(goal, briefing, failure_summary)
    if decision in ("stop", "ping"): break
truth = env.ground_truth()
log episode: tokens (from usage), cached_tokens, tool calls, verdicts, truth, decision, claim_ok
if brain: brain.curate(trace, verdicts, truth_free_summary)   # curator sees verdicts, NOT ground truth
```

Operator turn contract (in the system prompt, enforced by a JSON check): every message ends with `{"progress": [lo, hi], "infeasible": false}`. Monitor stops on `infeasible` or on `hi` collapsing below `lo` of three turns ago.

Monitor rules, in order: loop (same tool + same args + same result twice in a row) → `replan`; turn > 25 or tokens > budget → `stop`; verifier frame unavailable → `ping`; otherwise `continue`.

## 6. Ground truth and scoring (`eval/score.py`)

`in_bin(block) = |xy_block − xy_bin| < 0.035 and z < 0.03`. Success = red in red bin, blue in blue bin, green on the table outside both bins, nothing holding, gripper open.

Per episode record: `success`, `claimed_done`, `verifier_pass`, `false_done = claimed_done and not success`, `verifier_correct = (verifier_pass == success)`, `tool_calls`, `turns`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `stale_servings` (brain only: a retrieved fact whose `valid_to` should have been set given the gotcha schedule), `decision`.

## 7. Tests before any LLM (`tests/test_scripted.py`)

1. `test_reach`: home pose renders; `move_above(0.22, 0.06)` reaches within 1 cm.
2. `test_grasp`: scripted red block: above → lower → close → lift → `holding=True`.
3. `test_sort`: scripted full sort succeeds in `ground_truth()`.
4. `test_flaky`: with `flaky_gripper`, single close fails, double close holds.
5. `test_frames`: `look()` and `verifier_frame()` write jpegs ≤ 120 KB.

If 1–3 pass at the 45-minute mark, the loop is done. If not, take the fallbacks in §3 immediately rather than tuning.

## 8. Cut lines

- 30 min in, IK not converging → pose-table fallback.
- 45 min in, grasps dropping → weld fallback.
- 60 min in, anything still red → freeze the env as is, fix only what the scripted sort needs. The agents and the brain are the project; the sim only has to be honest and repeatable.
