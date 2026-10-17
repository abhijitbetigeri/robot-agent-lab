"""Lab bench simulation: SO-101 arm, sample tubes (blocks), racks (bins), two cameras.

Agents only ever see the five tools: scene_state, look, move_above, lower, lift, gripper.
ground_truth() and verifier_frame() are for the eval harness and the sealed verifier only.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_XML = HERE / "so101" / "lab.xml"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN = 1.4
GRIPPER_CLOSED = -0.10
HOVER_Z = 0.07
TUBE_H = 0.030
GRASP_Z = 0.020          # grasp centre height: above the tube midpoint (tube is 3 cm tall)
# Site frame facts (measured): finger axis = site +x; jaws close along site z; closed grasp centre = site + R @ GRASP_OFFSET
PARK_Q = np.array([0.0, -1.7, 1.6, 0.0, 0.0])   # folded back: out of the camera view, no floor contact
GRASP_OFFSET = np.array([0.001, 0.0, -0.002])   # measured: zero push on descent, holds on every block
IK_SEEDS = [np.array([0.0, 0.6, 1.0, -1.5, 0.0]), np.array([0.0, 1.2, 0.9, -1.6, 0.0]),
            np.array([0.0, 0.2, 1.5, -1.6, 0.0]), np.array([0.0, -0.5, 1.6, -1.0, 0.0])]
BLOCKS = ["red", "green", "blue"]
BINS = {"red": "bin_red", "blue": "bin_blue"}  # green has no rack on purpose
IN_BIN_RADIUS = 0.035
IN_BIN_MAX_Z = 0.05


@dataclass
class Gotchas:
    """Injected bench quirks the shared brain is supposed to learn."""
    flaky_gripper: bool = True      # first close per grasp attempt does nothing and reports holding=None
    bin_shift_from_episode: int = 3  # bin_blue moves +0.08 m in y from this episode on (0 = never)
    cam_offset_from_episode: int = 5  # operator_cam shifted 3 cm from this episode on (0 = never)
    verifier_cam_down: bool = False   # on-stage "unplug": verifier_frame() raises


@dataclass
class Trace:
    calls: list = field(default_factory=list)
    next_id: int = 1

    def add(self, tool: str, args: dict, result: dict) -> int:
        cid = self.next_id
        self.next_id += 1
        self.calls.append({"call_id": cid, "tool": tool, "args": args, "result": result, "t": time.time()})
        return cid


class LabEnv:
    def __init__(self, seed: int = 0, episode: int = 1, gotchas: Gotchas | None = None, frames_dir: str | Path = "runs/frames"):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.d = mujoco.MjData(self.m)
        self.d_ik = mujoco.MjData(self.m)  # scratch for IK
        self.rng = np.random.default_rng(seed)
        self.seed, self.episode = seed, episode
        self.g = gotchas or Gotchas()
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.trace = Trace()
        self._renderer = None

        self.site = self.m.site("gripperframe").id
        self.arm_dofs = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_qpos = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_ctrl = np.array([self.m.actuator(j).id for j in ARM_JOINTS])
        self.grip_ctrl = self.m.actuator("gripper").id
        self.grip_qpos = self.m.joint("gripper").qposadr[0]
        self.block_qpos = {c: self.m.joint(f"j_{c}").qposadr[0] for c in BLOCKS}
        self.block_geom = {c: self.m.geom(f"g_{c}").id for c in BLOCKS}
        self.bin_geom = {c: self.m.geom(n).id for c, n in BINS.items()}
        self.jaw_fixed = [i for i in range(self.m.ngeom) if self.m.geom(i).name.startswith("fixed_jaw")]
        self.jaw_moving = [i for i in range(self.m.ngeom) if self.m.geom(i).name.startswith("moving_jaw")]
        self.jnt_lo = self.m.jnt_range[[self.m.joint(j).id for j in ARM_JOINTS], 0]
        self.jnt_hi = self.m.jnt_range[[self.m.joint(j).id for j in ARM_JOINTS], 1]
        self.base_bin_pos = {c: self.m.geom_pos[g].copy() for c, g in self.bin_geom.items()}
        self.base_cam_pos = self.m.cam_pos[self.m.camera("operator_cam").id].copy()
        self._close_attempts = 0
        self._gripper_cmd = "open"
        self.on_step = None   # optional callback(env) invoked after every physics step (used by the demo recorder)
        self._view_renderer = None
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> dict:
        mujoco.mj_resetData(self.m, self.d)
        # gotchas that change the bench layout
        for c, g in self.bin_geom.items():
            self.m.geom_pos[g] = self.base_bin_pos[c]
        if self.g.bin_shift_from_episode and self.episode >= self.g.bin_shift_from_episode:
            self.m.geom_pos[self.bin_geom["blue"]] = self.base_bin_pos["blue"] + np.array([0.0, 0.08, 0.0])
        cam = self.m.camera("operator_cam").id
        self.m.cam_pos[cam] = self.base_cam_pos
        if self.g.cam_offset_from_episode and self.episode >= self.g.cam_offset_from_episode:
            self.m.cam_pos[cam] = self.base_cam_pos + np.array([0.03, 0.0, 0.0])
        # randomize tube positions a little (same seed → same layout)
        for c in BLOCKS:
            a = self.block_qpos[c]
            base = self.m.body(f"block_{c}").pos
            jitter = self.rng.uniform(-0.02, 0.02, size=2)
            self.d.qpos[a:a + 3] = [base[0] + jitter[0], base[1] + jitter[1], TUBE_H / 2 + 0.001]
            self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        # arm: home = hover over the bench centre, open gripper
        self.d.ctrl[self.grip_ctrl] = GRIPPER_OPEN
        self.d.qpos[self.grip_qpos] = GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)
        self._settle(200)
        self._park()
        self._close_attempts = 0
        self._gripper_cmd = "open"
        self.trace = Trace()
        return {"ok": True, "episode": self.episode}

    # ------------------------------------------------------------------ the six tools
    def park(self) -> dict:
        steps = self._park()
        res = {"ok": True, "steps": steps}
        res["call_id"] = self.trace.add("park", {}, res)
        return res

    def scene_state(self) -> dict:
        q = self.d.qpos[self.arm_qpos]
        res = {"joints_deg": {j: round(float(np.degrees(v)), 1) for j, v in zip(ARM_JOINTS, q)},
               "gripper": self._gripper_cmd, "sim_time_s": round(float(self.d.time), 2)}
        res["call_id"] = self.trace.add("scene_state", {}, res)
        return res

    def look(self) -> dict:
        """Operator camera (top-down). Returns detected tubes/racks in bench coordinates plus the image path.
        Detection is a plain colour-centroid detector; the camera can be biased by a gotcha, and the detector
        inherits that bias — which is the point."""
        from sim.perceive import detect
        path = self._snap("operator_cam", 512, 512, "op")
        cam = self.m.camera("operator_cam").id
        det = detect(str(path), cam_pos=self.base_cam_pos, fovy_deg=float(self.m.cam_fovy[cam]))  # NOMINAL pose: a nudged camera biases every detection
        res = {"tubes": det["tubes"], "racks": det["racks"], "image": str(path)}
        res["call_id"] = self.trace.add("look", {}, {"tubes": det["tubes"], "racks": det["racks"], "image": str(path)})
        return res

    def move_above(self, x: float, y: float) -> dict:
        err, steps = self._goto(np.array([x, y, HOVER_Z]))
        res = {"ok": err < 0.012, "reached_xy_err_m": round(float(err), 4), "steps": steps}
        res["call_id"] = self.trace.add("move_above", {"x": x, "y": y}, res)
        return res

    def lower(self) -> dict:
        p = self._grasp_centre()
        err, steps = self._goto(np.array([p[0], p[1], GRASP_Z]))
        res = {"ok": err < 0.012, "z_err_m": round(float(err), 4), "steps": steps, "touching": self._touching_any_block()}
        res["call_id"] = self.trace.add("lower", {}, res)
        return res

    def lift(self) -> dict:
        p = self._grasp_centre()
        err, steps = self._goto(np.array([p[0], p[1], HOVER_Z]))
        res = {"ok": err < 0.015, "steps": steps, "holding": self._holding()}
        res["call_id"] = self.trace.add("lift", {}, res)
        return res

    def gripper(self, cmd: str) -> dict:
        if cmd not in ("open", "close"):
            res = {"ok": False, "error": "cmd must be 'open' or 'close'"}
            res["call_id"] = self.trace.add("gripper", {"cmd": cmd}, res)
            return res
        if cmd == "open":
            self.d.ctrl[self.grip_ctrl] = GRIPPER_OPEN
            self._settle(250)
            self._close_attempts = 0
            self._gripper_cmd = "open"
            res = {"ok": True, "holding": False}
        else:
            self._close_attempts += 1
            if self.g.flaky_gripper and self._close_attempts == 1:
                # the sticky gripper: servo does not engage on the first command
                self._settle(50)
                res = {"ok": True, "holding": None}
            else:
                self.d.ctrl[self.grip_ctrl] = GRIPPER_CLOSED
                self._settle(300)
                self._gripper_cmd = "closed"
                res = {"ok": True, "holding": self._holding()}
        res["call_id"] = self.trace.add("gripper", {"cmd": cmd}, res)
        return res

    # ------------------------------------------------------------------ eval / verifier only
    def verifier_frame(self) -> str:
        if self.g.verifier_cam_down:
            raise RuntimeError("verifier camera unavailable")
        return str(self._snap("verifier_cam", 512, 512, "ver"))

    def ground_truth(self) -> dict:
        pos = {c: self.d.qpos[self.block_qpos[c]:self.block_qpos[c] + 3].copy() for c in BLOCKS}
        bins = {c: self.m.geom_pos[g][:2] for c, g in self.bin_geom.items()}
        where = {}
        for c in BLOCKS:
            where[c] = "table"
            for b, bxy in bins.items():
                if np.linalg.norm(pos[c][:2] - bxy) < IN_BIN_RADIUS and pos[c][2] < IN_BIN_MAX_Z:
                    where[c] = b
        success = (where["red"] == "red" and where["blue"] == "blue" and where["green"] == "table"
                   and not self._holding() and self._gripper_cmd == "open")
        return {"block_xy": {c: [round(float(v), 3) for v in pos[c][:2]] for c in BLOCKS},
                "block_in": where, "bins_xy": {c: [round(float(v), 3) for v in v2] for c, v2 in bins.items()},
                "holding": self._holding(), "gripper": self._gripper_cmd, "success": bool(success)}

    # ------------------------------------------------------------------ internals
    def _step(self):
        mujoco.mj_step(self.m, self.d)
        if self.on_step:
            self.on_step(self)

    def render_view(self, w: int = 960, h: int = 540, lookat=(0.20, 0.0, 0.03), distance=0.62, azimuth=150.0, elevation=-32.0):
        """Angled 'audience' camera for the demo recorder (MjvCamera, no XML change)."""
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(self.m, h, w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        self._view_renderer.update_scene(self.d, camera=cam)
        return self._view_renderer.render()

    def _park(self) -> int:
        self.d.ctrl[self.arm_ctrl] = PARK_Q
        for steps in range(1, 601):
            self._step()
            if np.max(np.abs(self.d.qpos[self.arm_qpos] - PARK_Q)) < 0.01 and steps > 150:
                break
        return steps

    def _settle(self, steps: int):
        for _ in range(steps):
            self._step()

    def _ik_once(self, q0: np.ndarray, target: np.ndarray, iters: int = 80, damping: float = 0.05) -> tuple[float, np.ndarray]:
        """Damped least squares. Target = world position of the closed-jaw grasp centre; finger axis (site x) points down."""
        d = self.d_ik
        d.qpos[:] = self.d.qpos
        q = q0.copy()
        down = np.array([0.0, 0.0, -1.0])
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        err_p = err_r = np.ones(3)
        for _ in range(iters):
            d.qpos[self.arm_qpos] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.site)
            R = d.site_xmat[self.site].reshape(3, 3)
            p_grasp = d.site_xpos[self.site] + R @ GRASP_OFFSET
            err_p = target - p_grasp
            err_r = np.cross(R[:, 0], down)
            J = np.vstack([jacp[:, self.arm_dofs], 0.5 * jacr[:, self.arm_dofs]])
            err = np.concatenate([err_p, 0.5 * err_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)
            if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 2e-2:
                break
        return float(np.linalg.norm(err_p) + 0.02 * np.linalg.norm(err_r)), q

    def _ik(self, target: np.ndarray) -> np.ndarray:
        """Multi-start IK: current pose plus canonical seeds facing the target."""
        pan = -np.arctan2(target[1], target[0])  # shoulder_pan sign: +pan turns the arm toward -y (measured)
        seeds = [self.d.qpos[self.arm_qpos].copy()]
        for s in IK_SEEDS:
            s = s.copy()
            s[0] = pan
            seeds.append(s)
        best = min((self._ik_once(s, target) for s in seeds), key=lambda t: t[0])
        return best[1]

    def _goto(self, target: np.ndarray, max_steps: int = 600) -> tuple[float, int]:
        q = self._ik(target)
        self.d.ctrl[self.arm_ctrl] = q
        steps = 0
        for steps in range(1, max_steps + 1):
            self._step()
            if np.max(np.abs(self.d.qpos[self.arm_qpos] - q)) < 0.005 and steps > 150:
                break
        err = float(np.linalg.norm(self._grasp_centre() - target))
        return err, steps

    def _grasp_centre(self) -> np.ndarray:
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site] + R @ GRASP_OFFSET

    def _contacts(self, geoms_a: list[int], geom_b: int) -> bool:
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 in geoms_a and c.geom2 == geom_b) or (c.geom2 in geoms_a and c.geom1 == geom_b):
                return True
        return False

    def _holding(self) -> bool:
        for c in BLOCKS:
            g = self.block_geom[c]
            if self._contacts(self.jaw_fixed, g) and self._contacts(self.jaw_moving, g):
                return True
        return False

    def _touching_any_block(self) -> bool:
        jaws = self.jaw_fixed + self.jaw_moving
        return any(self._contacts(jaws, self.block_geom[c]) for c in BLOCKS)

    def _snap(self, cam: str, w: int, h: int, tag: str) -> Path:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 768, 768)  # square: no aspect distortion on resize
        self._renderer.update_scene(self.d, camera=cam)
        img = Image.fromarray(self._renderer.render()).resize((w, h))
        path = self.frames_dir / f"ep{self.episode:02d}_{tag}_{self.trace.next_id:03d}.jpg"
        img.save(path, quality=80)
        return path

    # ------------------------------------------------------------------ tool schema for the operator
    @staticmethod
    def tool_schemas() -> list[dict]:
        f = lambda name, desc, props=None, req=None: {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": req or []}}}
        return [
            f("park", "Fold the arm back out of the camera's view. Call this before look() so the arm does not hide tubes."),
            f("scene_state", "Joint angles (deg), gripper command state, sim time. Does NOT include tube positions; use look()."),
            f("look", "Take a photo from the operator camera. Returns an image path the harness attaches as an image."),
            f("move_above", "Move the gripper to hover 7 cm above bench point (x, y) in metres, jaws pointing down.",
              {"x": {"type": "number"}, "y": {"type": "number"}}, ["x", "y"]),
            f("lower", "Descend straight down to grasp height at the current x,y."),
            f("lift", "Rise straight up to hover height at the current x,y."),
            f("gripper", "Open or close the jaws.", {"cmd": {"type": "string", "enum": ["open", "close"]}}, ["cmd"]),
        ]

    def call(self, name: str, args: dict) -> dict:
        fn = {"park": self.park, "scene_state": self.scene_state, "look": self.look, "move_above": self.move_above,
              "lower": self.lower, "lift": self.lift, "gripper": self.gripper}[name]
        return fn(**args)
