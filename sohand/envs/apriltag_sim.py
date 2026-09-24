"""Simulated ground-truth cube-pose sensor + a hardware-shareable tracker.

TWO CLASSES, ON PURPOSE
-----------------------
  TagTracker          Everything a *deployed* system does with detections:
                      per-tag cube-pose recovery, weighted fusion, outlier
                      gating, hold-when-blind, staleness, same-tag windowed
                      velocity. It knows nothing about MuJoCo or ground truth.
                      On hardware you feed it real detections; in simulation
                      AprilTagEstimator feeds it synthetic ones. SAME CODE,
                      so the policy's velocity/pose observation has the same
                      definition in both worlds. Import this file on the robot.

  AprilTagEstimator   Simulation only. Synthesises what a per-face pose sensor
                      would have produced from the true cube pose (geometric
                      visibility, occlusion, noise, dropout, frame rate,
                      latency), hands those "detections" to a TagTracker, and
                      returns the tracker's output. The name is kept for
                      import compatibility with the rest of the repo, but as
                      of this revision it no longer models an AprilTag/ArUco
                      DECODER (no pixel-size/focal-length model, no planar
                      pose-ambiguity flips) -- see point 9 below.

Nothing here is validated against real detector logs. Every number marked
"SET FROM YOUR HARDWARE" in `AprilTagCfg` is a placeholder until you measure it.

Cube: 6 cm edge (0.03 m half-extent), read from the compiled MuJoCo model at
runtime (`env.cube_half`) -- nothing here hardcodes it. The 1 cm figure that
used to appear in this file was the printed AprilTag DECAL size, which no
longer matters now that per-face pose is read from ground truth rather than
reconstructed from a decoded marker's apparent size.

CHANGES IN THIS REVISION
------------------------
(earlier revisions: per-tag windowed velocity, face-normal visibility, cube-
relative camera, centre-relative tag frames -- all kept.)

 1. GROUND-TRUTH LEAK REMOVED. The old code fell back to the TRUE cube pose on
    every step until the first detection ("bootstrap"). `_held_pos` was never
    set on that path, so it was not a one-frame event -- an episode that began
    blind was fed ground truth until a tag appeared. Now the tracker holds a
    NON-privileged prior (nominal cube centre, identity rotation) and reports
    `bootstrap=True` / rising staleness until a real detection arrives.
 2. CAMERA FRAME RATE. Real cameras run ~30 fps; control runs at 50 Hz. A new
    detection now arrives only on steps where a frame was captured (random
    phase per episode) and is delivered `latency` steps later. The tracker
    differences poses by CAPTURE time, exactly as it must on hardware.
 3. LATENCY delivers detections (not fused poses) after the delay, which is
    where the delay physically sits. Default range widened to 2-7 steps
    (40-140 ms); SET FROM YOUR HARDWARE.
 4. `n_visible` is now taken from the latest DELIVERED frame, consistent with
    the delayed pose. Before, it came from the undelayed instant.
 5. [SUPERSEDED BY POINT 9] Pixel-size-dependent detection probability and
    noise used to model a real AprilTag decoder. Removed.
 6. [SUPERSEDED BY POINT 9] Pose-ambiguity outlier flips (planar-marker
    artifact) used to be injected as a heavy tail. Removed -- there is no
    marker to flip anymore, and the tracker's outlier gating (fusion and
    velocity) is kept regardless, since a real system still needs it.
 7. CAMERA EXTRINSIC BIAS acts about the camera, so a rotation error also
    displaces distant points (lever arm) -- it used to be a pure additive
    offset plus an orientation-only rotation.
 8. OPTIONAL RAY-CAST OCCLUSION (env supplies `occluded_faces`): a tag is
    hidden if any non-cube geom is between the camera and it, not only if a
    finger presses that face.
 9. MARKER-DECODE MODEL REMOVED; GROUND-TRUTH-BASED SENSOR IN ITS PLACE. Each
    non-excluded, geometrically-visible face's TRUE pose (from MuJoCo, i.e.
    "what a perfectly-decoded marker would have reported") is now read
    directly, instead of being reconstructed through a synthetic focal-length
    / pixel-size / decode-probability / pose-flip model of an actual AprilTag
    detector. This is NOT a ground-truth leak: the realism that made this a
    noisy, imperfect sensor rather than an oracle is all still applied
    downstream of that read --  per-episode camera-calibration bias,
    per-detection Gaussian pose noise, a dropout probability, camera frame
    rate, and delivery latency -- and it is still fed through the same
    `TagTracker` bootstrap/hold/staleness/velocity-gating machinery. What is
    gone is only the physical-marker-decode simulation (tag print/mount
    tolerance, focal length, apparent pixel size, decode-probability falloff,
    depth bias from a mis-printed tag, and planar pose-ambiguity flips) --
    fields `tag_pos_err_m`, `tag_rot_err_rad`, `tag_size_err_frac`,
    `cam_focal_px`, `detect_px_50`, `detect_px_slope`, `nominal_distance_m`,
    `outlier_p`, `outlier_rot_rad` are removed from `AprilTagCfg` accordingly.
    Rationale: that model made visibility depend on a tiny (1 cm) decal's
    apparent size, which on this rig was frequently below the decode
    threshold -- `ep_tag_boot_frac`/`ep_tag_visible_mean` in training logs
    would stay high/low respectively, i.e. the policy was often blind and
    holding a stale prior, which starves it of the closed-loop pose feedback
    it needs to learn to rotate the cube at all. Swapping in a ground-truth
    read of the (much bigger, 6 cm) cube face removes that artificial
    bottleneck while keeping every other randomisation knob live.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sohand.rotations import q_from_axis_angle, quat_to_mat, so3_log


@dataclass(frozen=True)
class AprilTagCfg:
    # --- per-episode systematic error (drawn once at reset) --------------
    # Hand-measured camera mounts are usually worse than 4 mm / 1 deg.
    cam_extrinsic_pos_err_m: float = 0.008   # SET FROM YOUR HARDWARE
    cam_extrinsic_rot_err_rad: float = 0.035 # ~2 deg
    cam_pos_nominal: tuple = (0.0, -0.22, 0.30)   # looking down+in
    cam_relative_to_cube: bool = True
    cam_fov_cos: float = 0.35            # geometric visibility cone (~70 deg)

    # --- camera / detector (SET FROM YOUR HARDWARE) -----------------------
    cam_fps_hz: float = 30.0             # 0 => a fresh frame every control step

    # --- per-detection measurement noise ------------------------------------
    # No pixel-size/focal-length model (see module docstring, point 9): a
    # ground-truth-based sensor has no "apparent size" to degrade with, so
    # these are flat per-detection noise stds/probabilities rather than
    # something that scales with distance or obliqueness.
    detect_pos_noise_m: float = 0.0015
    detect_rot_noise_rad: float = 0.02
    detect_dropout_p: float = 0.03       # geometrically visible but missed

    # --- timing --------------------------------------------------------
    latency_steps_range: tuple = (2, 7)  # camera + detect + transport, SET FROM YOUR HARDWARE
    max_stale_steps: int = 30            # for normalising the staleness obs

    # --- occlusion by ray casting (env-side; see AmazingHandSpinEnv) --------
    raycast_occlusion: bool = True
    ray_tol_m: float = 0.003

    # --- tracker (shared with hardware) -----------------------------------
    # WINDOW/GATE RETUNED. At 10 steps the window rarely contained two frames
    # sharing the same visible tag (grip + a single fixed camera hide most
    # faces most of the time), so almost every velocity estimate fell back to
    # the shortest available baseline -- differencing two independently noisy
    # detections (~0.02 rad each) one frame apart puts the noise floor
    # ABOVE the true spin rate (0.02-0.8 rad/s). Longer window = more chances
    # to find a longer, lower-noise baseline (the tracker already prefers the
    # oldest match first); a looser gate stops that longer baseline's own
    # first real reading from being thrown out for "disagreeing" with a
    # filtered estimate that started at zero.
    vel_window_steps: int = 20
    vel_lpf: float = 0.5
    vel_blind_decay: float = 0.95
    # steps without a fresh rate before the held rate starts to decay
    vel_hold_steps: int = 6
    # per-tag rate above this is a flip/outlier, not spin (rad/s) ...
    vel_reject_rad_s: float = 4.0
    # ... and a per-tag rate this far from the CURRENT filtered rate is treated
    # as an outlier too (innovation gate). Three consecutive fully-gated frames
    # bypass the gate, so a genuine fast spin from rest cannot lock it out.
    vel_gate_rad_s: float = 1.6
    # with >=3 tags, drop any tag this far (rad) from the fused rotation
    fuse_reject_rad: float = 0.45


# ---------------------------------------------------------------------------
# Data passed from a detector (real or simulated) to the tracker
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One tag seen in one frame, already expressed in the hand/world frame the
    policy uses (i.e. after the camera extrinsics). `weight` is any positive
    confidence, e.g. 1 / (distance / nominal) or the detector's decision margin."""
    face: int
    R: np.ndarray        # tag orientation, hand frame
    p: np.ndarray        # tag position, hand frame
    weight: float = 1.0


@dataclass
class Frame:
    capture_t: float               # capture time in CONTROL-STEP units
    detections: list = field(default_factory=list)


def _weighted_rotation_average(rots, weights):
    """Cheap weighted rotation average: fine for the small angular spread
    between simultaneously-visible tags on one rigid cube.

    `weights` are normalized internally so the caller doesn't have to
    remember to: the axis-angle reconstruction below depends on the raw
    magnitude of `acc`, so passing in unnormalized weights (e.g. raw
    per-detection confidences that don't sum to 1) would scale the fused
    rotation by sum(weights) instead of averaging it."""
    weights = np.asarray(weights, dtype=float)
    w_sum = weights.sum()
    if w_sum > 1e-9:
        weights = weights / w_sum
    ref = rots[0]
    acc = np.zeros(3)
    for R, w in zip(rots, weights):
        acc += w * so3_log(R @ ref.T)
    d_theta = np.linalg.norm(acc)
    if d_theta < 1e-9:
        return ref
    return quat_to_mat(q_from_axis_angle(acc / d_theta, d_theta)) @ ref


class TagTracker:
    """Detections in, pose + windowed velocity + confidence out.

    Hardware use::

        tracker = TagTracker(tag_local_poses, cfg)
        tracker.reset(prior_pos=nominal_cube_centre)
        # every control tick (50 Hz), with whatever frames arrived since last tick:
        out = tracker.tick(now_steps, frames, dt)

    `now_steps` and `Frame.capture_t` must be on the same clock, in control-step
    units (seconds / dt). On hardware use the CAMERA'S capture timestamp for
    `capture_t`, not the arrival time -- velocity is differenced over capture
    time.

    `tag_local_poses[i] = (pos_3, R_3x3)` is the NOMINAL (as-designed) tag pose
    in the frame of the cube's geometric centre, indexed by face.
    """

    def __init__(self, tag_local_poses, cfg: AprilTagCfg = AprilTagCfg()):
        self.cfg = cfg
        self.local = tag_local_poses

    def reset(self, prior_pos=None, prior_R=None):
        self._prior_pos = np.zeros(3) if prior_pos is None else np.array(prior_pos, float)
        self._prior_R = np.eye(3) if prior_R is None else np.array(prior_R, float)
        self._held_pos = self._prior_pos.copy()
        self._held_R = self._prior_R.copy()
        self._ever_detected = False
        self._t0 = None
        self._last_detect_now = None
        self._n_visible = 0
        self._hist = deque()
        self._omega_f = np.zeros(3)
        self._vlin_f = np.zeros(3)
        self._last_rate_now = None
        self._gated_run = 0        # consecutive frames with tags but no accepted rate

    # -- per frame ---------------------------------------------------------
    def _fuse(self, dets):
        c = self.cfg
        tags = {}
        for d in dets:
            loc_pos, loc_R = self.local[d.face]
            R_est = d.R @ loc_R.T
            p_est = d.p - R_est @ loc_pos
            tags[d.face] = (R_est, p_est, float(d.weight))
        keys = list(tags)
        if len(keys) >= 3:
            rots = [tags[k][0] for k in keys]
            ref = _weighted_rotation_average(rots, np.array([tags[k][2] for k in keys]))
            keep = [k for k in keys
                    if np.linalg.norm(so3_log(tags[k][0] @ ref.T)) <= c.fuse_reject_rad]
            if len(keep) >= 2:
                tags = {k: tags[k] for k in keep}
        w = np.array([tags[k][2] for k in tags])
        w = w / w.sum()
        pos = np.sum(np.array([tags[k][1] for k in tags]) * w[:, None], axis=0)
        R = _weighted_rotation_average([tags[k][0] for k in tags], w)
        return pos, R, tags

    def _ingest(self, frame: Frame, now):
        if not frame.detections:
            self._hist.append({"t": frame.capture_t, "tags": {}})
            self._n_visible = 0
        else:
            pos, R, tags = self._fuse(frame.detections)
            self._held_pos, self._held_R = pos, R
            self._ever_detected = True
            self._last_detect_now = now
            self._n_visible = len(tags)
            self._hist.append({"t": frame.capture_t, "tags": tags})
        win = max(int(self.cfg.vel_window_steps), 1)
        while self._hist and frame.capture_t - self._hist[0]["t"] > win:
            self._hist.popleft()

    def _velocity_from_newest(self, dt):
        """Rate from the newest frame against the OLDEST frame in the window that
        shares a plausible tag with it. Same-tag differencing cancels every
        static per-tag / per-camera bias; a per-tag rate above
        `vel_reject_rad_s` is treated as a pose flip and ignored."""
        c = self.cfg
        new = self._hist[-1]
        if not new["tags"]:
            return None
        for old in list(self._hist)[:-1]:                  # oldest first
            span = new["t"] - old["t"]
            if span < 1e-6:
                continue
            w_acc, om_acc, v_acc = 0.0, np.zeros(3), np.zeros(3)
            for i in new["tags"].keys() & old["tags"].keys():
                Rn, pn, wn = new["tags"][i]
                Ro, po, _ = old["tags"][i]
                om = so3_log(Rn @ Ro.T)
                rate_i = om / (span * dt)
                if (np.linalg.norm(rate_i) > c.vel_reject_rad_s
                        or (self._gated_run < 3
                            and np.linalg.norm(rate_i - self._omega_f) > c.vel_gate_rad_s)):
                    continue
                om_acc += wn * om
                v_acc += wn * (pn - po)
                w_acc += wn
            if w_acc > 0.0:
                return om_acc / (w_acc * span * dt), v_acc / (w_acc * span * dt)
        return None

    # -- per control tick ----------------------------------------------------
    def tick(self, now, frames, dt):
        """`frames`: frames DELIVERED since the previous tick (possibly none)."""
        c = self.cfg
        if self._t0 is None:
            self._t0 = now
        for f in frames:
            self._ingest(f, now)
        if frames:
            rate = self._velocity_from_newest(dt)
            # after 3 consecutive fully-gated frames the innovation gate is
            # bypassed, so a genuine fast spin from rest cannot be locked out
            self._gated_run = self._gated_run + 1 if (rate is None and self._hist[-1]["tags"]) else 0
            if rate is not None:
                a = c.vel_lpf
                self._omega_f = (1.0 - a) * self._omega_f + a * rate[0]
                self._vlin_f = (1.0 - a) * self._vlin_f + a * rate[1]
                self._last_rate_now = now
            else:
                self._omega_f *= c.vel_blind_decay
                self._vlin_f *= c.vel_blind_decay
        else:
            # between frames: hold; only decay once the rate is genuinely old
            ref = self._last_rate_now if self._last_rate_now is not None else self._t0
            if now - ref > c.vel_hold_steps:
                self._omega_f *= c.vel_blind_decay
                self._vlin_f *= c.vel_blind_decay

        since = now - (self._last_detect_now if self._last_detect_now is not None
                       else self._t0)
        return {
            "pos": self._held_pos.copy(), "R": self._held_R.copy(),
            "angvel": self._omega_f.copy(), "linvel": self._vlin_f.copy(),
            "n_visible": self._n_visible,
            "stale_frac": min(max(since, 0) / c.max_stale_steps, 1.0),
            "bootstrap": not self._ever_detected,
        }


# ---------------------------------------------------------------------------
# Simulation-side detector synthesis
# ---------------------------------------------------------------------------
class AprilTagEstimator:
    """One instance per env; call `reset()` every episode, `step()` every
    control step. Ground truth is passed in only so this class can synthesise
    what a camera would have seen."""

    def __init__(self, face_normals, cfg: AprilTagCfg = AprilTagCfg(),
                 tag_local_poses=None):
        """`tag_local_poses`, when given, IS the nominal layout: (pos_3, R_3x3)
        per face, in the frame of the cube's GEOMETRIC CENTRE. The fallback
        `_build_nominal_layout` is not guaranteed to match any scene.xml."""
        self.cfg = cfg
        self.face_normals = np.asarray(face_normals, dtype=float)
        self._nominal_local = tag_local_poses
        self.cam_pos = np.array(cfg.cam_pos_nominal, dtype=float)
        self._nominal = False
        self._dr_scale = 1.0
        self._cam_fps_eff = cfg.cam_fps_hz
        self.tracker = None

    # -- geometry ---------------------------------------------------------
    def _build_nominal_layout(self, cube_half: float):
        """FALLBACK ONLY."""
        world_up = np.array([0.0, 0.0, 1.0])
        layout = []
        for n in self.face_normals:
            ref = world_up if abs(np.dot(n, world_up)) < 0.9 else np.array([1.0, 0.0, 0.0])
            t1 = np.cross(ref, n)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            pos = n * cube_half            # face centre; no decal to inset from
            rot = np.stack([t1, t2, n], axis=1)
            layout.append((pos, rot))
        self._nominal_local = layout

    def reset(self, rng: np.random.Generator, cube_half: float, nominal: bool = False,
              cam_target=None, dr_scale: float = 1.0):
        """`nominal=True` (deterministic eval) zeroes every random draw and
        captures every step with zero latency -- but never reveals ground
        truth: it still only sees geometrically visible, unoccluded tags.

        `dr_scale` in [0, 1]: a CURRICULUM knob, not a way to skip the
        pipeline -- it is still read every step at every scale.
        At 1.0 every number is exactly `AprilTagCfg` (full realism). Below
        that, every noise std, the dropout probability and the latency
        range shrink toward 0 continuously, and the effective camera frame
        rate rises toward "every control step" -- an easier, still-imperfect
        version of the same pipeline, not ground truth. Ramp this from your
        train loop once `ep_revolutions`/`success` moves off zero at a given
        scale; 1.0 is the only setting that matches the real camera.

        `cam_target`: world position of the nominal cube centre; required when
        `cfg.cam_relative_to_cube`. It is also the tracker's non-privileged
        prior for the blind start."""
        c = self.cfg
        self._nominal = nominal
        self._dr_scale = float(np.clip(dr_scale, 0.0, 1.0))
        # Frame rate is the one knob curriculum makes EASIER by going ABOVE
        # spec (more frames = faster feedback), converging down to the real
        # `cam_fps_hz` at scale 1.0 -- never below it, so this never claims a
        # worse camera exists than the real one.
        self._cam_fps_eff = c.cam_fps_hz + (60.0 - c.cam_fps_hz) * (1.0 - self._dr_scale)
        if self._nominal_local is None:
            self._build_nominal_layout(cube_half)

        if nominal:
            self._cam_bias_pos = np.zeros(3)
            self._cam_bias_R = np.eye(3)
            self._latency_steps = 0
            self._cam_acc = 0.0
        else:
            s = self._dr_scale
            self._cam_bias_pos = rng.normal(0.0, c.cam_extrinsic_pos_err_m * s, 3)
            self._cam_bias_R = quat_to_mat(q_from_axis_angle(
                rng.normal(size=3) + 1e-9,
                float(rng.normal(0.0, c.cam_extrinsic_rot_err_rad * s))))
            lat_lo = int(round(c.latency_steps_range[0] * s))
            lat_hi = max(lat_lo, int(round(c.latency_steps_range[1] * s)))
            self._latency_steps = int(rng.integers(lat_lo, lat_hi + 1))
            self._cam_acc = float(rng.uniform(0.0, 1.0))     # random frame phase

        offset = np.array(c.cam_pos_nominal, dtype=float)
        if c.cam_relative_to_cube:
            if cam_target is None:
                raise ValueError("AprilTagCfg.cam_relative_to_cube=True needs "
                                 "reset(..., cam_target=<nominal cube centre>)")
            base = np.asarray(cam_target, dtype=float) + offset
        else:
            base = offset
        self._cam_nominal = base                        # where the estimator BELIEVES it is
        self.cam_pos = base + self._cam_bias_pos        # where the camera really is

        self._pending = deque()
        self._t = 0
        self._rng = rng
        self.tracker = TagTracker(self._nominal_local, c)
        self.tracker.reset(prior_pos=None if cam_target is None else cam_target)

    # -- per-step synthesis -------------------------------------------------
    def _tag_world_pose(self, i, cube_pos_true, cube_R_true):
        """TRUE world pose of face `i`'s tracked point, straight from the
        simulator -- ground truth, before any of the sensor randomisation in
        `_detect` is applied."""
        loc_pos, loc_R = self._nominal_local[i]
        true_pos = cube_pos_true + cube_R_true @ loc_pos
        true_R = cube_R_true @ loc_R
        return true_pos, true_R

    def tag_positions_world(self, cube_pos_true, cube_R_true):
        """True world position of every tag centre (for env-side ray casting)."""
        return [self._tag_world_pose(i, cube_pos_true, cube_R_true)[0]
                for i in range(len(self.face_normals))]

    def face_view_cosines(self, cube_pos_true, cube_R_true):
        """cos(angle between each face's outward normal and the direction to the
        camera). Diagnostic; a tag is geometrically visible above
        `cfg.cam_fov_cos` (before occlusion and dropout)."""
        out = np.zeros(len(self.face_normals))
        for i in range(len(self.face_normals)):
            tag_pos, _ = self._tag_world_pose(i, cube_pos_true, cube_R_true)
            to_cam = self.cam_pos - tag_pos
            d = np.linalg.norm(to_cam)
            out[i] = np.dot(cube_R_true @ self.face_normals[i], to_cam / max(d, 1e-9))
        return out

    def _frame_due(self, dt):
        c = self.cfg
        if self._nominal or self._cam_fps_eff <= 0.0:
            return True
        self._cam_acc += self._cam_fps_eff * dt
        if self._cam_acc >= 1.0:
            self._cam_acc -= 1.0
            return True
        return False

    def _detect(self, cube_pos_true, cube_R_true, excluded_faces):
        """What a ground-truth-based pose sensor returns for ONE captured
        frame: each geometrically-visible, non-excluded face's TRUE pose,
        run through camera-calibration bias, per-detection noise and a
        dropout probability -- but NOT through a synthetic AprilTag decoder
        (no focal length / pixel size / decode-probability / pose-flip
        model; see module docstring point 9). Every face that clears the
        FOV cone and the dropout draw is equally trustworthy, so `weight`
        is flat instead of scaling with apparent size."""
        c = self.cfg
        dets = []
        for i in range(len(self.face_normals)):
            if i in excluded_faces:              # pressed by a finger / ray-occluded
                continue
            true_pos, true_R = self._tag_world_pose(i, cube_pos_true, cube_R_true)
            to_cam = self.cam_pos - true_pos
            d = float(np.linalg.norm(to_cam))
            if d < 1e-6:
                continue
            cosang = float(np.dot(cube_R_true @ self.face_normals[i], to_cam / d))
            if cosang < c.cam_fov_cos:
                continue                         # camera physically can't see this face
            if not self._nominal and self._rng.random() < c.detect_dropout_p * self._dr_scale:
                continue                         # otherwise-visible face, detector missed it

            if self._nominal:
                meas_pos, meas_R = true_pos, true_R
            else:
                s = self._dr_scale
                meas_pos = true_pos + self._rng.normal(0.0, c.detect_pos_noise_m * s, 3)
                rot_noise = quat_to_mat(q_from_axis_angle(
                    self._rng.normal(size=3) + 1e-9,
                    float(self._rng.normal(0.0, c.detect_rot_noise_rad * s))))
                meas_R = true_R @ rot_noise
            # Camera extrinsic error acts ABOUT the real camera (lever arm
            # included). The real camera sits at cam_pos = cam_nominal +
            # cam_bias_pos with orientation cam_bias_R relative to nominal;
            # reconstructing world position/orientation using the NOMINAL
            # extrinsics (what an uncalibrated detector believes) requires
            # the INVERSE (transpose, since these are rotations) of that
            # bias, applied about the real camera position -- not the bias
            # itself applied about the nominal position.
            meas_pos = (self._cam_nominal
                        + self._cam_bias_R.T @ (meas_pos - self._cam_nominal
                                                - self._cam_bias_pos))
            meas_R = self._cam_bias_R.T @ meas_R
            dets.append(Detection(face=i, R=meas_R, p=meas_pos, weight=1.0))
        return dets

    def step(self, cube_pos_true, cube_R_true, touched_faces, dt, occluded_faces=()):
        """Returns dict: pos, R, angvel, linvel, n_visible, stale_frac, bootstrap
        (True until the first real detection is delivered -- during that time
        pos/R are the non-privileged prior, NOT ground truth)."""
        now = self._t
        if self._frame_due(dt):
            excluded = set(touched_faces) | set(occluded_faces)
            frame = Frame(capture_t=float(now),
                          detections=self._detect(cube_pos_true, cube_R_true, excluded))
            self._pending.append((now + self._latency_steps, frame))
        delivered = []
        while self._pending and self._pending[0][0] <= now:
            delivered.append(self._pending.popleft()[1])
        out = self.tracker.tick(float(now), delivered, dt)
        self._t += 1
        return out