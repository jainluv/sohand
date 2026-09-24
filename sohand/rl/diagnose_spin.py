"""Merged SoHand RL Utilities: Diagnostics, Policy Export, Hardware Runtime, Parity Verification, and Stress Testing.

Subcommands:
    diagnose     - Run environment diagnostics to analyze grasp, perception, commands, and contacts.
    export       - Export actor model to ONNX, TorchScript, and generate metadata JSON.
    check-parity - Verify hardware deploy runtime parity against the training environment.
    stress-test  - Evaluate policy robustness outside the nominal training distribution.

Usage:
    python sohand_tools.py diagnose --mode random --noise
    python sohand_tools.py export --model runs/spin/models/best_model.zip --out deploy/
    python sohand_tools.py check-parity --meta deploy/policy_meta.json
    python sohand_tools.py stress-test --model runs/spin/models/best_model.zip
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

import numpy as np

# ==============================================================================
# DEPLOY RUNTIME (deploy_runtime.py)
# ==============================================================================

OBS_ORDER = ("jpos", "jvel", "last_action", "prev_action", "filtered_ctrl",
             "tilt_cos", "tilt_vec", "phase4", "angvel", "linvel",
             "center_offset", "axis", "tag_conf")


def tag_local_poses(meta):
    return [(np.asarray(p, float), np.asarray(R, float)) for p, R in meta["tag_local_poses"]]


def meta_tag_cfg(meta):
    c = dict(meta["apriltag_cfg"])
    for k in ("cam_pos_nominal", "latency_steps_range"):      # JSON turns tuples into lists
        if k in c:
            c[k] = tuple(c[k])
    return c


def cube_spin_features(R, k_hat):
    """Copy of spin.cube_spin_features (kept identical; parity-tested)."""
    axes = R.T
    dots = axes @ k_hat
    i = int(np.argmax(np.abs(dots)))
    a_up = axes[i] * np.sign(dots[i])
    tilt_cos = float(np.clip(np.dot(a_up, k_hat), -1.0, 1.0))
    tilt_vec = a_up - tilt_cos * k_hat
    j = (i + 1) % 3
    e1 = np.array([k_hat[1], -k_hat[2], k_hat[0]])
    e1 = e1 - np.dot(e1, k_hat) * k_hat
    n1 = np.linalg.norm(e1)
    e1 = e1 / n1 if n1 > 1e-8 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(k_hat, e1)
    a_p = axes[j] - np.dot(axes[j], k_hat) * k_hat
    psi = float(np.arctan2(np.dot(a_p, e2), np.dot(a_p, e1)))
    return tilt_cos, tilt_vec.astype(np.float32), np.array(
        [np.cos(4 * psi), np.sin(4 * psi)], dtype=np.float32)


class ActionShaper:
    """Policy action in [-1, 1] -> filtered joint command (fraction of half-range).

    Mirrors AmazingHandSpinEnv.step: target = clip(grasp + a * band), one-pole
    low-pass, then a per-step slew limit. The band is centred on the grasp
    COMMAND (`anchor_grasp_on_command`), not on where blocked fingers stopped.
    """

    def __init__(self, meta, ctrl_rate_scale: float = 1.0):
        self.band = float(meta["grasp_band_frac"])
        self.lpf = float(meta["action_lpf"])
        # training drew this per episode in +-dr_ctrl_rate; on hardware use the
        # nominal (1.0) unless you have measured the servo's sustained slew.
        self.rate = float(meta["max_ctrl_rate_frac"]) * ctrl_rate_scale
        self.grasp = np.zeros(len(meta["ctrl_mid"]), np.float32)
        self.filtered = np.zeros_like(self.grasp)

    def reset(self, grasp_frac):
        self.grasp = np.asarray(grasp_frac, np.float32).copy()
        self.filtered = self.grasp.copy()

    def step(self, action):
        act = np.clip(np.nan_to_num(np.asarray(action, np.float32), nan=0.0), -1.0, 1.0)
        target = np.clip(self.grasp + act * self.band, -1.0, 1.0)
        lpf = self.lpf * target + (1.0 - self.lpf) * self.filtered
        delta = np.clip(lpf - self.filtered, -self.rate, self.rate)
        self.filtered = (self.filtered + delta).astype(np.float32)
        return self.filtered


def joint_targets(meta, filtered):
    """Radians to send to the servos (position mode)."""
    mid, half = np.asarray(meta["ctrl_mid"]), np.asarray(meta["ctrl_half"])
    return mid + np.asarray(filtered) * half


def grasp_frac_command(meta, close_frac=None):
    """The closure COMMAND (action-fraction units) the settle drove the servos to."""
    mid, half = np.asarray(meta["ctrl_mid"]), np.asarray(meta["ctrl_half"])
    lo, hi = np.asarray(meta["ctrl_lo"]), np.asarray(meta["ctrl_hi"])
    cf = meta["hand_close_frac"] if close_frac is None else close_frac
    q_closed = np.clip(mid + meta["close_sign"] * cf * half, lo, hi)
    return np.clip((q_closed - mid) / half, -1.0, 1.0)


def grasp_ramp(meta, close_frac=None):
    """Joint-target sequence (one row per 20 ms tick) that reproduces the sim's
    open -> closed ramp. After the last row HOLD the closed command; do not
    re-centre on the measured (blocked) position -- that removes the squeeze."""
    mid, half = np.asarray(meta["ctrl_mid"]), np.asarray(meta["ctrl_half"])
    lo, hi = np.asarray(meta["ctrl_lo"]), np.asarray(meta["ctrl_hi"])
    cf = meta["hand_close_frac"] if close_frac is None else close_frac
    q_open = np.clip(mid + meta["close_sign"] * meta["grasp_open_frac"] * half, lo, hi)
    q_closed = np.clip(mid + meta["close_sign"] * cf * half, lo, hi)
    n = int(meta["settle_ctrl_steps"])
    return np.stack([q_open + (i + 1) / n * (q_closed - q_open) for i in range(n)])


def build_obs(meta, jpos_raw, jvel_raw, last_action, prev_action, filtered, track):
    """`track` is the dict returned by TagTracker.tick()."""
    mid, half = np.asarray(meta["ctrl_mid"]), np.asarray(meta["ctrl_half"])
    k_hat = np.asarray(meta["k_hat"], float)
    jpos = np.clip((np.asarray(jpos_raw) - mid) / half, -1.0, 1.0)
    jvel = np.clip(np.asarray(jvel_raw) / meta["jvel_scale"], -1.0, 1.0)
    tilt_cos, tilt_vec, phase4 = cube_spin_features(track["R"], k_hat)
    angvel = np.clip(track["angvel"] / meta["angvel_obs_scale"], -3.0, 3.0)
    linvel = np.clip(track["linvel"] / meta["linvel_obs_scale"], -3.0, 3.0)
    offset = np.clip((track["pos"] - np.asarray(meta["nominal_cube_center"])) / 0.05, -3.0, 3.0)
    tag_conf = np.array([track["n_visible"] / 6.0, track["stale_frac"]], dtype=np.float32)
    return np.concatenate([
        jpos, jvel, last_action, prev_action, filtered,
        [tilt_cos], tilt_vec, phase4, angvel, linvel, offset, k_hat, tag_conf,
    ]).astype(np.float32)


# ==============================================================================
# DIAGNOSE SPIN (diagnose_spin.py)
# ==============================================================================

FACE_NAMES = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]


def _flag(ok: bool, msg: str) -> str:
    return f"  [{'OK   ' if ok else 'CHECK'}] {msg}"


def diagnose_spin_main(args) -> None:
    from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
    from sohand.hand import FACE_NORMALS
    from sohand.paths import CUBE_SCENE
    from sohand.rotations import so3_log

    cfg_kw = {}
    if args.close_frac is not None:
        cfg_kw["hand_close_frac"] = args.close_frac
    if args.no_anchor_grasp:
        cfg_kw["anchor_grasp_on_command"] = False
    env = AmazingHandSpinEnv(
        model_path=args.scene or CUBE_SCENE,
        randomize=args.noise, sensor_noise=args.noise,
        cfg=make_spin_cfg(**cfg_kw) if cfg_kw else make_spin_cfg())

    model = None
    if args.mode == "policy":
        from stable_baselines3 import SAC
        if not args.model:
            raise SystemExit("--mode policy needs --model")
        model = SAC.load(args.model, device="cpu")
    rng = np.random.default_rng(args.seed)

    # ---- geometry -----------------------------------------------------------
    print(f"cube half-extent {env.cube_half * 1000:.1f} mm   body-origin->centre "
          f"{np.round(env.cube_local_center * 1000, 1)} mm   spin axis {env.cfg.spin_axis}")
    env.reset(seed=args.seed)
    est = env._tag_estimator
    cos = est.face_view_cosines(env._cube_center_world(), env._cube_R())
    thr = est.cfg.cam_fov_cos
    vis_faces = [FACE_NAMES[i] for i in range(6) if cos[i] >= thr]
    print("camera view cosine per face (need >= %.2f):  " % thr
          + "  ".join(f"{FACE_NAMES[i]}={cos[i]:+.2f}" for i in range(6)))
    print(f"  camera at {np.round(est.cam_pos, 3)}   faces the camera can see at the "
          f"start pose: {vis_faces or 'NONE'}")

    # ---- episodes ------------------------------------------------------------
    R = dict(valid=[], tries=[], close=[], touch0=[])
    S = {k: [] for k in ("vis", "stale", "n_touch", "act", "dq", "dctrl", "true_w", "est_w",
                         "track", "pressed_top", "faces", "boot", "ray_occ")}
    ep_rot_true, ep_rot_est, dropped = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep,
                           options={"start_face": ep % 6} if args.cycle_faces else None)
        R["valid"].append(env._reset_valid)
        R["tries"].append(env._reset_tries)
        R["close"].append(env._reset_close_used)
        R["touch0"].append(int(env._fingers_touching_mask().sum()))
        Rp = env._cube_R().copy()
        qprev = env.data.qpos[env.qposids].copy()
        cprev = env.filtered_ctrl.copy()
        rot_t = rot_e = 0.0
        for t in range(args.steps):
            if args.mode == "zero":
                a = np.zeros(env.action_space.shape, np.float32)
            elif args.mode == "random":
                a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)

            Rc = env._cube_R()
            w = so3_log(Rc @ Rp.T) / env.dt
            Rp = Rc.copy()
            e = env._tag_est
            q = env.data.qpos[env.qposids].copy()
            faces = env._touched_faces()
            S["vis"].append(e["n_visible"]); S["stale"].append(e["stale_frac"])
            S["n_touch"].append(info["n_touch"])
            S["act"].append(float(np.mean(np.abs(a))))
            S["dq"].append(float(np.mean(np.abs(q - qprev))))
            S["dctrl"].append(float(np.mean(np.abs(env.filtered_ctrl - cprev))))
            S["true_w"].append(float(w @ env.k_hat)); S["est_w"].append(float(e["angvel"] @ env.k_hat))
            S["track"].append(float(np.degrees(np.linalg.norm(so3_log(e["R"] @ Rc.T)))))
            # the cube face currently pointing at world +Z (the face the camera sees best)
            top_face = int(np.argmax((Rc @ np.asarray(FACE_NORMALS).T)[2]))
            S["pressed_top"].append(top_face in faces)
            S["faces"].extend(faces)
            S["boot"].append(float(e["bootstrap"])); S["ray_occ"].append(len(env._last_occ))
            qprev, cprev = q, env.filtered_ctrl.copy()
            rot_t += float(w @ env.k_hat) * env.dt
            rot_e += float(e["angvel"] @ env.k_hat) * env.dt
            if term or trunc:
                break
        ep_rot_true.append(rot_t); ep_rot_est.append(rot_e); dropped.append(float(term))

    A = {k: np.asarray(v, float) for k, v in S.items()}

    # ---- A. grasp ---------------------------------------------------------
    print("\nA. GRASP")
    print(f"  valid resets {np.mean(R['valid']):.0%}   attempts/reset {np.mean(R['tries']):.2f}   "
          f"closure used {np.mean(R['close']):.2f}   fingers touching at start {np.mean(R['touch0']):.2f}   "
          f"dropped {np.mean(dropped):.0%}")
    print(_flag(np.mean(R["valid"]) >= 0.9,
                "resets start from a verified grasp" if np.mean(R["valid"]) >= 0.9 else
                "many resets fail validation -> episodes start ungrasped. Change --close-frac / scene"))

    # ---- B. perception ----------------------------------------------------
    seen = float(np.mean(A["vis"] > 0))
    print("\nB. PERCEPTION (what the policy's cube-pose input is built from)")
    print(f"  tags visible per step {A['vis'].mean():.2f}   steps with >=1 tag {seen:.0%}   "
          f"stale_frac mean {A['stale'].mean():.2f} max {A['stale'].max():.2f}")
    if S["faces"]:
        cnt = np.bincount(np.asarray(S["faces"], int), minlength=6)
        print("  cube-body faces held by a finger (steps): " + "  ".join(f"{FACE_NAMES[i]}={cnt[i]}" for i in range(6)))
    print(f"  tags hidden by ray-cast occlusion per step {A['ray_occ'].mean():.2f}   "
          f"steps still on the PRIOR (no tag ever delivered) {A['boot'].mean():.1%}")
    if env._occluder_counts:
        top = sorted(env._occluder_counts.items(), key=lambda kv: -kv[1])[:6]
        print("  geoms that hide tags (steps): " + "  ".join(f"{n}={c}" for n, c in top)
              + "   <- each must be something really between camera and cube")
    print(f"  fraction of steps a finger presses the face that points UP (world +Z): {A['pressed_top'].mean():.0%}")
    print(f"  pose error vs truth {A['track'].mean():.1f} deg mean  ({np.percentile(A['track'], 95):.1f} p95)")
    sig = np.sqrt(np.mean(A["true_w"] ** 2)); noise = np.sqrt(np.mean((A["est_w"] - A["true_w"]) ** 2))
    print(f"  spin rate about k:  true rms {sig:.3f} rad/s   estimate error rms {noise:.3f} rad/s   "
          f"rotation over the run  true {np.mean(ep_rot_true):+.2f} rad  perceived {np.mean(ep_rot_est):+.2f} rad")
    print(_flag(seen >= 0.8, f"a tag is detected on {seen:.0%} of steps" if seen >= 0.8 else
                f"the policy is BLIND on {1 - seen:.0%} of steps -> its pose input is a stale hold. "
                "Look at which faces are pressed and at the camera cosines above"))
    print(_flag(A["boot"].mean() < 0.05, "policy is seeing real detections almost from step 0"
                if A["boot"].mean() < 0.05 else
                "episodes start BLIND (prior pose only) -- camera cannot see the start pose"))
    print(_flag(A["track"].mean() < 15.0, "estimate follows the true orientation" if A["track"].mean() < 15.0
                else "estimate is far from the true orientation -- check tag layout / frames"))
    print(_flag(noise < max(1.0 * sig, 0.08),
                "angular-velocity estimate noise is small vs the signal" if noise < max(1.0 * sig, 0.08)
                else "angular-velocity estimate is mostly noise -- the policy cannot see progress"))

    # ---- C. command -------------------------------------------------------
    print("\nC. COMMAND (mode = %s)" % args.mode)
    print(f"  mean |raw action| {A['act'].mean():.3f}   mean |d filtered command|/step {A['dctrl'].mean():.4f}   "
          f"mean |d joint pos|/step {A['dq'].mean():.5f} rad")
    if args.mode != "zero":
        cmd_ok = A["act"].mean() > 0.05
        print(_flag(cmd_ok, "policy commands motion" if cmd_ok else
                    "policy outputs ~0: it is genuinely not trying (reward / observation problem)"))
        if cmd_ok:
            follow = A["dq"].mean() / max(A["dctrl"].mean() * float(np.mean(env.ctrl_half)), 1e-9)
            print(_flag(follow > 0.2, f"joints follow the command (ratio {follow:.2f})" if follow > 0.2 else
                        "commands are issued but joints barely move: LPF/slew limit or blocked fingers"))

    # ---- D. contact ------------------------------------------------------
    two = float(np.mean(A["n_touch"] >= 2))
    print("\nD. CONTACT")
    print(f"  fingers on cube: mean {A['n_touch'].mean():.2f}   >=2 fingers on {two:.0%} of steps")
    print(_flag(two >= 0.7, "cube is held" if two >= 0.7 else
                "cube is not held by >=2 fingers most of the time -- fix the grasp before the reward"))
    print("\nThe first CHECK above is the thing to fix. Nothing about SAC or the reward matters "
          "until A-D are OK.")


# ==============================================================================
# EXPORT POLICY (export_policy.py)
# ==============================================================================

def build_meta(env) -> dict:
    from sohand.hand import (ACTUATORS, JOINTS, FACE_NORMALS, GRASP_OPEN_FRAC,
                             JVEL_SCALE, SETTLE_CTRL_STEPS)
    from sohand.envs.spin import (ANGVEL_OBS_SCALE, LINVEL_OBS_SCALE,
                                  SPIN_OBS_DIM, SPIN_OBS_SLICES)
    c = env.cfg
    return {
        "dt": float(env.dt), "control_hz": float(1.0 / env.dt),
        "obs_dim": int(SPIN_OBS_DIM), "obs_slices": dict(SPIN_OBS_SLICES),
        "joints": list(JOINTS), "actuators": list(ACTUATORS),
        "ctrl_lo": env.ctrl_lo.tolist(), "ctrl_hi": env.ctrl_hi.tolist(),
        "ctrl_mid": env.ctrl_mid.tolist(), "ctrl_half": env.ctrl_half.tolist(),
        "jvel_scale": float(JVEL_SCALE),
        "angvel_obs_scale": float(ANGVEL_OBS_SCALE),
        "linvel_obs_scale": float(LINVEL_OBS_SCALE),
        "k_hat": env.k_hat.tolist(), "spin_axis": c.spin_axis,
        "nominal_cube_center": env.nominal_cube_center.tolist(),
        "cube_half": float(env.cube_half),
        "close_sign": float(env.close_sign),
        "hand_close_frac": float(c.hand_close_frac),
        "grasp_open_frac": float(GRASP_OPEN_FRAC),
        "settle_ctrl_steps": int(SETTLE_CTRL_STEPS),
        "grasp_band_frac": float(c.grasp_band_frac),
        "action_lpf": float(c.action_lpf),
        "max_ctrl_rate_frac": float(c.max_ctrl_rate_frac),
        "action_dropout_p_train": float(c.action_dropout_p),
        "apriltag_cfg": asdict(c.apriltag),
        "tag_local_poses": [[np.asarray(p).tolist(), np.asarray(R).tolist()]
                            for p, R in env._tag_estimator._nominal_local],
        "face_normals": np.asarray(FACE_NORMALS).tolist(),
    }


def export_policy_main(args) -> None:
    import torch as th
    from stable_baselines3 import SAC

    os.makedirs(args.out, exist_ok=True)
    model = SAC.load(args.model, device="cpu")
    actor = model.policy.actor.eval()
    obs_dim = int(model.observation_space.shape[0])
    lo, hi = model.action_space.low, model.action_space.high
    if not (np.allclose(lo, -1.0) and np.allclose(hi, 1.0)):
        raise SystemExit(f"action space is [{lo.min()}, {hi.max()}], expected [-1, 1]")

    class DeterministicActor(th.nn.Module):
        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, obs):
            return self.a(obs, deterministic=True)       # tanh(mean), no sampling

    net = DeterministicActor(actor).eval()
    dummy = th.zeros(1, obs_dim)

    ts_path = os.path.join(args.out, "policy.pt")
    th.jit.trace(net, dummy).save(ts_path)

    onnx_path = os.path.join(args.out, "policy.onnx")
    onnx_ok = True
    try:
        kw = dict(input_names=["obs"], output_names=["action"], opset_version=17,
                  dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
        try:
            th.onnx.export(net, dummy, onnx_path, dynamo=False, **kw)
        except TypeError:                      # older torch has no `dynamo` kwarg
            th.onnx.export(net, dummy, onnx_path, **kw)
    except Exception as e:                     # noqa: BLE001
        onnx_ok = False
        print(f"[export] ONNX export failed ({e}); TorchScript export is still valid.")

    # ---- parity: SB3 predict vs TorchScript vs ONNX -------------------------
    rng = np.random.default_rng(0)
    obs = np.clip(rng.normal(0.0, 0.6, (512, obs_dim)), -3.0, 3.0).astype(np.float32)
    ref, _ = model.predict(obs, deterministic=True)
    with th.no_grad():
        ts = th.jit.load(ts_path)(th.from_numpy(obs)).numpy()
    err_ts = float(np.abs(ts - ref).max())
    print(f"[export] TorchScript vs SB3 predict: max |diff| = {err_ts:.2e}")
    bad = err_ts > 1e-4
    if onnx_ok:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"obs": obs})[0]
            err_ox = float(np.abs(out - ref).max())
            print(f"[export] ONNX        vs SB3 predict: max |diff| = {err_ox:.2e}")
            bad = bad or err_ox > 1e-4
        except ImportError:
            print("[export] onnxruntime not installed -- ONNX parity not checked")
    if bad:
        raise SystemExit("[export] PARITY FAILED -- do not deploy these files")

    # ---- meta: needs the env (scene, ctrl ranges, tag layout) ------------------
    from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
    from sohand.paths import CUBE_SCENE
    kw = {}
    if args.close_frac is not None:
        kw["hand_close_frac"] = args.close_frac
    env = AmazingHandSpinEnv(model_path=args.scene or CUBE_SCENE, randomize=False,
                             sensor_noise=False, cfg=make_spin_cfg(**kw))
    meta = build_meta(env)
    if meta["obs_dim"] != obs_dim:
        raise SystemExit(f"policy expects {obs_dim} obs but env builds {meta['obs_dim']}: "
                         "the model was trained on a different observation layout")
    meta["source_model"] = os.path.abspath(args.model)
    with open(os.path.join(args.out, "policy_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export] wrote policy.pt, policy.onnx, policy_meta.json to {args.out}")


# ==============================================================================
# CHECK DEPLOY PARITY (check_deploy_parity.py)
# ==============================================================================

def check_deploy_parity_main(args) -> None:
    from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
    from sohand.envs.apriltag_sim import AprilTagCfg, TagTracker
    from sohand.paths import CUBE_SCENE

    kw = {"hand_close_frac": args.close_frac} if args.close_frac is not None else {}
    env = AmazingHandSpinEnv(model_path=args.scene or CUBE_SCENE, randomize=False,
                             sensor_noise=False, cfg=make_spin_cfg(**kw))
    meta = (json.load(open(args.meta)) if args.meta
            else json.loads(json.dumps(build_meta(env))))

    # record every tracker tick the env makes (including the one inside reset)
    log = []
    orig_tick = TagTracker.tick

    def spy(self, now, frames, dt):
        out = orig_tick(self, now, frames, dt)
        log.append((now, list(frames), dt, out))
        return out
    TagTracker.tick = spy

    obs, _ = env.reset(seed=args.seed, options={"start_face": 0})
    shaper = ActionShaper(meta)
    shaper.reset(env.grasp_frac)
    worst_grasp = float(np.abs(grasp_frac_command(meta) - env.grasp_frac).max())
    rng = np.random.default_rng(args.seed)
    a_prev = np.zeros(8, np.float32)
    e_ctrl = e_obs = 0.0
    a = np.zeros(8, np.float32)
    for t in range(args.steps):
        # temporally-correlated random action, so the slew limit is exercised
        a = np.clip(0.9 * a + 0.5 * rng.normal(size=8), -1, 1).astype(np.float32)
        filt = shaper.step(a)
        obs, r, term, trunc, info = env.step(a)
        e_ctrl = max(e_ctrl, float(np.abs(filt - env.filtered_ctrl).max()))
        mine = build_obs(meta, env.data.qpos[env.qposids], env.data.qvel[env.qvelids],
                         env.last_action, env.prev_action, shaper.filtered, env._tag_est)
        e_obs = max(e_obs, float(np.abs(mine - obs).max()))
        if term or trunc:
            break
    TagTracker.tick = orig_tick

    # independent tracker, fed the identical frames
    tr = TagTracker(tag_local_poses(meta), AprilTagCfg(**meta_tag_cfg(meta)))
    tr.reset(prior_pos=meta["nominal_cube_center"])
    e_trk = 0.0
    for now, frames, dt, ref in log:
        out = tr.tick(now, frames, dt)
        for k in ("pos", "R", "angvel", "linvel"):
            e_trk = max(e_trk, float(np.abs(out[k] - ref[k]).max()))
        e_trk = max(e_trk, abs(out["n_visible"] - ref["n_visible"]),
                    abs(out["stale_frac"] - ref["stale_frac"]))

    rows = [("grasp command (close_frac -> frac)", worst_grasp),
            ("ActionShaper vs env.filtered_ctrl", e_ctrl),
            ("build_obs vs env observation", e_obs),
            ("independent TagTracker vs env tracker", e_trk)]
    ok = True
    for name, err in rows:
        good = err < 1e-5
        ok &= good
        print(f"  [{'OK   ' if good else 'FAIL '}] {name:42s} max|diff| = {err:.2e}")
    print("PARITY " + ("PASSED" if ok else "FAILED -- do not deploy"))
    raise SystemExit(0 if ok else 1)


# ==============================================================================
# STRESS TEST (stress_test.py)
# ==============================================================================

def _cfg(base, tag=None, **kw):
    ap = replace(base.apriltag, **tag) if tag else base.apriltag
    return replace(base, apriltag=ap, **kw)


def conditions(base):
    return [
        ("training distribution", _cfg(base)),
        ("friction x2 wider", _cfg(base, dr_friction=0.60, dr_rolling_friction=0.60)),
        ("cube mass x2.5 wider", _cfg(base, dr_mass=0.50)),
        ("servo gain/damping wider", _cfg(base, dr_gain=0.35, dr_damping=0.50)),
        ("slower servo slew", _cfg(base, dr_ctrl_rate=0.45)),
        ("softer/stiffer contact", _cfg(base, dr_contact_softness=0.50)),
        ("encoder bias/noise x2.5", _cfg(base, encoder_bias_rad=0.015, noise_jpos_rad=0.025,
                                          joint_assembly_offset_rad=0.025)),
        ("long latency (camera+encoder)", _cfg(base, tag=dict(latency_steps_range=(8, 12)),
                                                encoder_latency_steps=(2, 4))),
        ("camera 10 fps", _cfg(base, tag=dict(cam_fps_hz=10.0))),
        ("worse camera calibration", _cfg(base, tag=dict(cam_extrinsic_pos_err_m=0.02,
                                                         cam_extrinsic_rot_err_rad=0.07))),
        ("small tags (low resolution)", _cfg(base, tag=dict(cam_focal_px=800.0))),
        ("many misses + flips", _cfg(base, tag=dict(detect_dropout_p=0.15, outlier_p=0.05))),
        ("combined (moderate)", _cfg(base, dr_friction=0.45, dr_mass=0.35, dr_gain=0.25,
                                     tag=dict(latency_steps_range=(4, 9), cam_fps_hz=15.0,
                                              cam_extrinsic_pos_err_m=0.012,
                                              detect_dropout_p=0.08, outlier_p=0.03))),
    ]


def stress_test_main(args) -> None:
    from stable_baselines3 import SAC
    from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
    from sohand.paths import CUBE_SCENE

    model = SAC.load(args.model, device="cpu")
    base = make_spin_cfg(**({"hand_close_frac": args.close_frac} if args.close_frac is not None else {}))
    print(f"{'condition':34s} {'rev':>6s} {'ok&held':>8s} {'drop':>6s} {'reset_ok':>9s}")
    for name, cfg in conditions(base):
        env = AmazingHandSpinEnv(model_path=args.scene or CUBE_SCENE, randomize=True,
                                 sensor_noise=True, cfg=cfg)
        revs, drops, valid = [], [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done, info = False, {}
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            revs.append(info["ep_revolutions"]); drops.append(info["ep_dropped"])
            valid.append(info["ep_reset_valid"])
        revs, drops = np.array(revs), np.array(drops)
        ok = np.mean((revs >= cfg.success_revolutions) & (drops == 0))
        print(f"{name:34s} {revs.mean():6.2f} {ok:8.2f} {drops.mean():6.2f} {np.mean(valid):9.2f}")
        env.close()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SoHand RL Diagnostics, Export, Parity, and Stress Test Suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: diagnose
    p_diag = subparsers.add_parser("diagnose", help="Diagnose environment issues (grasp, perception, command, contact)")
    p_diag.add_argument("--scene", default=None)
    p_diag.add_argument("--mode", choices=["zero", "random", "policy"], default="zero")
    p_diag.add_argument("--model", default=None)
    p_diag.add_argument("--episodes", type=int, default=6)
    p_diag.add_argument("--steps", type=int, default=300)
    p_diag.add_argument("--seed", type=int, default=0)
    p_diag.add_argument("--noise", action="store_true",
                        help="training-style randomisation + sensor noise (default: nominal)")
    p_diag.add_argument("--cycle-faces", action="store_true", help="start on faces 0..5 in turn")
    p_diag.add_argument("--close-frac", type=float, default=None)
    p_diag.add_argument("--no-anchor-grasp", action="store_true")

    # Subcommand: export
    p_exp = subparsers.add_parser("export", help="Export trained policy actor for hardware execution")
    p_exp.add_argument("--model", required=True)
    p_exp.add_argument("--out", default="deploy")
    p_exp.add_argument("--scene", default=None)
    p_exp.add_argument("--close-frac", type=float, default=None,
                       help="MUST match the --close-frac the model was trained with")

    # Subcommand: check-parity
    p_par = subparsers.add_parser("check-parity", help="Prove hardware deploy runtime reproduces training env")
    p_par.add_argument("--meta", default=None)
    p_par.add_argument("--scene", default=None)
    p_par.add_argument("--close-frac", type=float, default=None)
    p_par.add_argument("--steps", type=int, default=300)
    p_par.add_argument("--seed", type=int, default=0)

    # Subcommand: stress-test
    p_str = subparsers.add_parser("stress-test", help="Score a trained policy outside its training distribution")
    p_str.add_argument("--model", required=True)
    p_str.add_argument("--scene", default=None)
    p_str.add_argument("--close-frac", type=float, default=None)
    p_str.add_argument("--episodes", type=int, default=20)
    p_str.add_argument("--seed", type=int, default=5000)

    args = parser.parse_args()

    if args.command == "diagnose":
        diagnose_spin_main(args)
    elif args.command == "export":
        export_policy_main(args)
    elif args.command == "check-parity":
        check_deploy_parity_main(args)
    elif args.command == "stress-test":
        stress_test_main(args)