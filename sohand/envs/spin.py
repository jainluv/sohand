import os
import warnings
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
import mujoco
from gymnasium import spaces

from sohand.envs.mujoco_env import MujocoEnv
from sohand.envs.apriltag_sim import AprilTagEstimator, AprilTagCfg
from sohand.hand import (
    ACTUATORS, JOINTS, TIP_SITES, N_JOINTS, N_FINGERS,
    JVEL_SCALE, REACH_NORM, FACE_NORMALS,
    HAND_CLOSE_FRAC, HAND_CLOSE_JITTER, GRASP_OPEN_FRAC, CLOSE_PROBE_FRAC,
    SETTLE_CTRL_STEPS,
)
from sohand.paths import CUBE_SCENE
from sohand.rotations import (
    qmul, q_from_axis_angle, q_from_vecs, quat_to_mat, so3_log,
)

# NOTE ON POSE SOURCE: `cfg.pose_source` picks where the cube pose in the
# observation comes from.
#   "gt"       (default) the true MuJoCo cube quaternion/position, passed
#              through `GTPoseSensor` -- a deliberately generic sensor model
#              (per-step noise, per-episode bias, latency, hold-last dropout).
#              No camera geometry, no detector model. The AprilTag pipeline is
#              not constructed and no rays are cast.
#   "apriltag" the old `AprilTagEstimator` path, unchanged.
# Either way the observation is built ONLY from that sensor's output plus the
# noisy/latent joint sensors; the raw `_cube_R()`/`_cube_center_world()` feed
# the sensor, the drop detector and the reward, never `_get_obs` directly.


# Angular-velocity normalisation for the observation. The reachable range is
# ~0.1-0.5 rad/s, so dividing by 10 (as the face-target env did) would squash
# the entire signal into +-0.05 of the observation range.
ANGVEL_OBS_SCALE = 2.0
LINVEL_OBS_SCALE = 0.5

# Named spin axes, in the world frame. The hand base is fixed to the world in
# this scene, so world == hand frame; the distinction only matters once the
# hand is mounted on a moving wrist, at which point k_hat must be rotated into
# the world frame before it is compared against the tracked cube angular
# velocity.
SPIN_AXES = {
    "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
    "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
}


def _rot_from_vec(v):
    """Rodrigues: rotation matrix for a rotation vector (small-angle safe)."""
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v, dtype=float) / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


@dataclass(frozen=True)
class PoseSensorCfg:
    """Generic cube-pose sensor error model for `pose_source="gt"`.

    Deliberately NOT tied to any particular sensor. These are stand-ins for
    "some quaternion+position source with ordinary error"; on hardware, log the
    real source against a reference and pull these numbers toward it. All noise
    scales with `dr_scale`, so the curriculum ramps them like everything else.
    """
    quat_noise_rad: float = 0.006      # per-step orientation jitter (~0.34 deg)
    quat_bias_rad: float = 0.015       # per-episode fixed orientation error (~0.9 deg)
    pos_noise_m: float = 0.001         # per-step position jitter
    pos_bias_m: float = 0.003          # per-episode fixed position error
    latency_steps: tuple = (0, 2)      # control steps (20 ms each), drawn per episode
    dropout_p: float = 0.01            # per-step chance a hold-last burst starts
    dropout_len: tuple = (1, 6)        # burst length in control steps, inclusive
    angvel_lpf: float = 0.25           # one-pole LPF on differenced angular velocity
    linvel_lpf: float = 0.25
    stale_window: int = 10             # held steps that saturate `stale_frac`


class GTPoseSensor:
    """Ground-truth cube pose -> what the policy is allowed to see.

    Orientation is corrupted as R_meas = exp(noise) @ R_bias @ R_true (rotation
    error, so it stays a valid rotation), position additively. Measurements go
    through a latency queue; a dropout freezes the delivered pose (hold-last),
    as a real tracker does when it loses the object. Velocities are finite-
    differenced from the DELIVERED poses over the true elapsed time between
    fresh frames, then low-passed -- the same thing a deployed tracker must do,
    so the policy learns on a velocity signal with realistic noise and lag.
    Same output keys as `AprilTagEstimator.step` that `_get_obs` consumes.
    """

    def __init__(self, cfg: PoseSensorCfg = PoseSensorCfg()):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.nominal = True
        self.s = 1.0
        self.reset(self.rng, np.eye(3), np.zeros(3), nominal=True, dr_scale=1.0)

    def _measure(self, R, pos):
        c, s = self.cfg, self.s
        if self.nominal:
            return R.copy(), pos.copy()
        noise = _rot_from_vec(self.rng.normal(0.0, c.quat_noise_rad * s, 3))
        Rm = noise @ self._R_bias @ R
        pm = pos + self._pos_bias + self.rng.normal(0.0, c.pos_noise_m * s, 3)
        return Rm, pm

    def reset(self, rng, R0, pos0, nominal=False, dr_scale=1.0):
        c = self.cfg
        self.rng, self.nominal, self.s = rng, bool(nominal), float(dr_scale)
        s = self.s
        if self.nominal:
            self._R_bias, self._pos_bias, n_lat = np.eye(3), np.zeros(3), 0
        else:
            self._R_bias = _rot_from_vec(rng.normal(0.0, c.quat_bias_rad * s, 3))
            self._pos_bias = rng.normal(0.0, c.pos_bias_m * s, 3)
            lo = int(round(c.latency_steps[0] * s))
            hi = max(lo, int(round(c.latency_steps[1] * s)))
            n_lat = int(rng.integers(lo, hi + 1))
        first = self._measure(np.asarray(R0), np.asarray(pos0))
        self._queue = deque([first] * (n_lat + 1), maxlen=n_lat + 1)
        self._out = first
        self._hold_left = 0
        self._held_run = 0            # consecutive held steps so far
        self._steps_since_fresh = 1
        self._angvel = np.zeros(3)
        self._linvel = np.zeros(3)

    def step(self, R_true, pos_true, dt):
        c, s = self.cfg, self.s
        self._queue.append(self._measure(R_true, pos_true))
        delayed = self._queue[0]

        held = False
        if not self.nominal:
            if self._hold_left > 0:
                self._hold_left -= 1
                held = True
            elif self.rng.random() < c.dropout_p * s:
                lo, hi = c.dropout_len
                self._hold_left = int(self.rng.integers(lo, hi + 1)) - 1
                held = True

        if held:
            self._held_run += 1
            self._steps_since_fresh += 1
        else:
            k = self._steps_since_fresh
            R_new, p_new = delayed
            R_old, p_old = self._out
            w = so3_log(R_new @ R_old.T) / (k * dt)
            v = (p_new - p_old) / (k * dt)
            self._angvel = (1 - c.angvel_lpf) * self._angvel + c.angvel_lpf * w
            self._linvel = (1 - c.linvel_lpf) * self._linvel + c.linvel_lpf * v
            self._out = delayed
            self._held_run = 0
            self._steps_since_fresh = 1

        R_o, p_o = self._out
        return {
            "R": R_o, "pos": p_o,
            "angvel": self._angvel.copy(), "linvel": self._linvel.copy(),
            "n_visible": 0.0 if held else 6.0,      # -> tag_conf[0]: fresh-frame flag
            "stale_frac": min(self._held_run / max(c.stale_window, 1), 1.0),
            "bootstrap": 0.0,
        }


@dataclass(frozen=True)
class SpinCfg:
    # --- episode ---------------------------------------------------------
    max_steps: int = 1000                 # 20 s at 50 Hz, matching Hora's 400 @ 20 Hz
    spin_axis: str = "+Z"                 # measured to be the only well-driven axis
    randomize_axis_sign: bool = False     # True -> also train the -Z direction

    # --- drop detection --------------------------------------------------
    # Unchanged from the face-target env: measured maxima under 1000-step
    # random-action episodes were 0.033 m lateral / 0.004 m vertical.
    z_drop_m: float = 0.035
    xy_drop_m: float = 0.055
    drop_persist_steps: int = 3

    # --- start-state randomisation ---------------------------------------
    pos_jitter_m: float = 0.004
    yaw_jitter_rad: float = float(np.radians(180.0))   # spin phase is irrelevant
    tilt_jitter_rad: float = float(np.radians(10.0))
    randomize_start_face: bool = True

    # A reset is only accepted if the settle actually produced a grasp. The
    # previous env accepted whatever the settle happened to leave, which made
    # the start state a large uncontrolled variance source -- it is what
    # destroyed levels 1-2 of the reverse curriculum (measured: an intended
    # 40-80 deg offset came out as 90.4 +- 3.1 deg because the settle tipped
    # the cube back onto a stable face).
    min_start_contacts: int = 2
    max_start_drift_m: float = 0.015
    max_reset_tries: int = 6
    # What happens when NO attempt produces a valid grasp. The old loop simply
    # fell through with whatever the last attempt left, so the episode started
    # from an ungrasped or already-displaced cube and nothing recorded it.
    #   - the closure is nudged between attempts: +step when too few fingers
    #     touch, -step when the cube was shoved too far (`reset_close_step`);
    #   - the BEST attempt (most contacts, least drift) is re-run and kept,
    #     not the last one;
    #   - `ep_reset_valid` is reported every episode so it is visible in the
    #     logs, and `strict_reset=True` raises instead of continuing (use it
    #     in preflight / diagnostics, not in a long training run).
    reset_close_step: float = 0.05
    strict_reset: bool = False

    # Evaluation (randomize=False) starts are DETERMINISTIC: no yaw/tilt/
    # position/closure jitter, and the start face cycles 0,1,2,... per reset so
    # a fixed number of eval episodes covers every face equally. Before, the
    # `randomize` flag did not reach `_place_and_settle`, so "deterministic"
    # evaluation still began from a random face, yaw, tilt and offset.
    # A `start_face` reset option always wins.

    # Anchor the action band on the grasp COMMAND, not on where the fingers
    # ended up. During settle the servos were commanded to `qpos_closed` and
    # the cube blocked them short of it -- that gap IS the grip force. The old
    # code then set filtered_ctrl to the blocked *position*, so on step 1 the
    # commanded target dropped to where the fingers already were and the
    # squeeze vanished. Do-nothing then showed ~0.6 of the contact-loss
    # penalty per step (fewer than 2 fingers on the cube 63% of the time).
    anchor_grasp_on_command: bool = True

    # Normal force above which a finger counts as touching (N). It gates reset
    # validation, tag occlusion, contact-loss and finger-idle -- so log its
    # effect (`ep_touch_mean`) before trusting any of them.
    contact_force_thresh: float = 0.015

    # How far the settle curls the fingers. A config field rather than the
    # imported constant because the right value depends on where the cube is:
    # at the v2 geometry the kinematic sweep puts the fingertips closest at
    # ~0.0-0.15, and leaving it at 0.50 makes the settle fail its own grasp
    # validation on all six attempts, every episode.
    hand_close_frac: float = HAND_CLOSE_FRAC

    # --- action mapping (unchanged; calibrated against the real servos) ---
    grasp_band_frac: float = 0.50
    action_lpf: float = 0.42
    # 0.08 frac/step = 5.6 rad/s of commanded joint speed. Measured joint
    # velocity under runs 1-2 was 2.3 rad/s mean and 3.3 p95, i.e. the policy
    # spends much of its time near the limit, which is what reads as frantic.
    # 0.05 = 3.5 rad/s: still well above what the task needs at a capped
    # 0.8 rad/s cube rate, and closer to what the real Feetech SCS0009
    # sustains under load rather than its no-load figure (0.1 s/60 deg @ 6 V
    # = 10.5 rad/s no-load; 3.5 rad/s is ~1/3 of that, a sensible loaded/
    # sagging-battery figure for a 2.3 kg.cm micro servo). Confirm on the
    # bench and pass the measured value as ctrl_rate_scale at deploy.
    max_ctrl_rate_frac: float = 0.05

    # --- reward ----------------------------------------------------------
    # WEIGHTS ARE MEASURED, NOT CHOSEN. A calibration sweep runs do-nothing,
    # random and a CEM-optimised rotator gait through this env and reports each
    # term's UNWEIGHTED per-episode sum. The first pass (2026-08-26) rejected a
    # set of plausible-looking Hora-derived weights outright:
    #
    #       term        do-nothing   uniform random   CEM rotator
    #       rot_clipped      0.75            2.20          4.51
    #       offaxis          2.58          202.6         505.1
    #       pose_sq          0.002         157.6         681.7
    #       work_sq          0.41        23527.0       14805.4
    #       torque_sq        0.84         2730.6        1779.2
    #       act_rate_sq      0.00         5228.9          33.2
    #       contact_loss   621.8            647.0         679.6
    #
    # Three of those terms are *anti-task* on this hand: pose, work and offaxis
    # each charge the rotator 100-1000x what they charge a frozen policy,
    # because on an 8-DoF hand whose entire action band is +-0.5 around the
    # grasp, moving away from the grasp pose IS the task. Hora can afford a
    # -0.3 pose penalty because a 16-DoF Allegro gaits *around* a canonical
    # grasp; this hand cannot. And contact_loss came out at ~0.63 for every
    # policy alike -- a constant offset carrying no gradient, so it is off.
    #
    # Invariant, re-verified after every weight change: a rotator must score
    # strictly above do-nothing. That is the single thing all three previous
    # runs got wrong.

    # The rotation term is the *increment*, w_rot * omega.k_hat * dt, so the
    # episode sum is exactly w_rot x (radians turned). No clipping artefact, no
    # farmable oscillation, and the return has a unit: one full revolution is
    # worth w_rot * 2pi = 50.
    w_rot: float = 8.0
    # Was 2.0, a pure spike guard, which left the rotation reward linear all the
    # way up and told the policy that faster is always better with no ceiling.
    # Run 2 took that literally: 1.586 rad/s, cube airborne 15.6% of the time,
    # only 1.22 fingers in contact, dropping 42.5% of episodes. Capping the
    # PAID rate means anything above `angvel_clip` earns nothing extra while
    # still paying the work and torque penalties, so a deliberate 0.8 rad/s
    # beats a frantic 1.6. Hora clips at 0.5 for the same reason.
    # 0.8 rad/s sustained = 2.5 revolutions per 20 s episode.
    angvel_clip: float = 0.8

    # Measured: the instantaneous per-control-step angular velocity is ~0.5
    # rad/s of contact-solver jitter riding on ~0.02-0.10 rad/s of actual spin,
    # a signal-to-noise ratio of about 1:25. A one-pole low-pass at ~2 Hz
    # removes the jitter and leaves the spin. It is linear with unit DC gain,
    # so the telescoping identity above survives up to one boundary term, and
    # it is exactly what a deployed pose tracker has to do anyway.
    angvel_lpf: float = 0.25

    # Off-axis tumbling, on the *filtered* angular velocity. Second calibration
    # pass: even filtered, the rotator's off-axis rate is 0.34 rad/s against a
    # mean on-axis rate of 0.0004 rad/s. That is not noise -- it is how this
    # hand turns the cube. With three 2-DoF fingers and a thumb there is no
    # clean spin available; the cube is rolled from face to face, and every
    # roll is off-axis motion. Hora and AnyRotate can charge -0.3 here because
    # a 16-DoF Allegro really can spin an object cleanly about one axis.
    # Weighted at 0.02 this term alone cost the rotator -6.79 against a +3.20
    # rotation reward. Kept at a whisper: a 5% nudge toward cleaner rotation,
    # never a reason not to rotate. Raise it for a polish phase once the task
    # is being solved, not before.
    w_offaxis: float = 0.0005
    offaxis_clip: float = 2.0

    # Cube-centre velocity, finite-differenced. NOT qvel[0:3]: that is the
    # velocity of the body *origin*, which sits 3.5 cm from the cube's centre,
    # so pure rotation at 0.5 rad/s shows up in it as 1.8 cm/s of phantom
    # translation and the penalty would charge the policy for rotating.
    w_linvel: float = 0.005
    w_drift: float = 0.01         # sustained displacement from the start pose

    # Hora uses -0.3 here. On this hand it is anti-task (see the table above).
    w_pose: float = 0.0
    w_work: float = 5e-6          # (sum tau * qdot)^2
    w_torque: float = 3e-5        # ||tau||^2
    # 0.004 was the original calibration, but measured at 6k steps it charges an
    # exploring policy -20.3 per episode against a do-nothing policy's 0.0 --
    # i.e. freezing would outscore exploring, which is precisely the failure
    # that produced three dead runs. 0.002 keeps twice the old deterrent
    # (-10 exploring, ~-1.3 for a converged gait) without inverting the
    # incentive. The slew limit above is the honest lever for "slower": it is a
    # physical constraint rather than a reward the policy can trade against.
    w_action_rate: float = 0.002  # command reversals: -0.03 for a smooth gait,
                                  # -5.2 for uniform random. A thrash deterrent
                                  # that costs a real gait essentially nothing.
    # Every weight above is set so that the penalties together come to ~19% of
    # what the rotation term pays a working gait -- a tax, not a competitor.
    # Measured episode totals at these weights (1000 steps):
    #   CEM rotator  +2.58     do-nothing  +0.10
    #   small random -0.90     uniform random -5.50
    # Privileged simulator-only grasp shaping. These are NOT observations.
    w_contact_loss: float = 0.005
    contact_min_fingers: int = 2

    # Privileged simulator-only fingertip envelope. Only penalise excessive
    # separation; do not reward minimum distance, which could encourage freezing.
    tip_safe_extra_m: float = 0.008
    tip_distance_clip_m: float = 0.025
    w_tip_distance: float = 0.01

    # Measured on the trained policies: finger1 bears force on 24% of steps
    # (run 1) or 29% (run 2) while finger2 manages 56%/41%. The geometry change
    # narrowed that gap but by lowering everyone, not raising finger1 --
    # fingers-in-contact fell 1.58 -> 1.22 and the cube spends 15.6% of the
    # episode airborne.
    #
    # This charges for the LEAST engaged finger being idle, using a ~1 s
    # exponential average of each finger's contact so it measures a gait rather
    # than an instant. It is a penalty, never a bonus, so the invariant holds:
    # a policy that grips with all four and does not rotate still scores ~0,
    # not more. Bounded by construction at -w_finger_idle per step.
    #
    # WEIGHT LOWERED 0.02 -> 0.005. At 0.02 the term is worth up to -20 per
    # 1000-step episode, and it is charged whenever ANY one finger is not on
    # the cube -- which, per the measurements above, is the normal state of
    # finger1 (24-29% contact). Measured on a do-nothing policy it alone costs
    # ~-19/episode. A rotator turning at a realistic 0.05-0.1 rad/s earns
    # 8 x (1-2 rad) = +8..16 for the whole episode, so the idle tax was larger
    # than the thing it was meant to be a whisper next to, and the cheapest way
    # to raise the return was to clamp every finger down and stay still. The
    # cap is now -5/episode (~2.5 rad of rotation), which still pushes toward
    # engaging all fingers but cannot outvote the task. Restore the old value
    # with `--w-finger-idle 0.02` to A/B it; `ep_r_finger` in the logs is the
    # number to watch.
    w_finger_idle: float = 0.005
    finger_ema: float = 0.02      # ~1 s window at 50 Hz

    # Terminal. One revolution is worth +50, so a drop costs a fifth of a turn
    # -- enough to matter, nowhere near enough to make freezing the safe play.
    # The real deterrent is termination: a drop forfeits the rest of the
    # episode's rotation reward.
    drop_penalty: float = 08.0
    terminate_on_drop: bool = True

    # --- reporting -------------------------------------------------------
    # "Success" is now a real, interpretable quantity: a full revolution about
    # the target axis inside one episode. Partial bars are logged too so the
    # learning curve is legible long before the first full turn.
    success_revolutions: float = 1.0

    # --- domain randomisation --------------------------------------------
    dr_friction: float = 0.30
    dr_mass: float = 0.20
    dr_damping: float = 0.25
    dr_gain: float = 0.15
    dr_ctrl_rate: float = 0.20
    dr_rolling_friction: float = 0.30    # geom_friction[:,2]; was left un-jittered
    dr_link_mass: float = 0.08           # hand/finger body mass+inertia; cube uses dr_mass
    dr_contact_softness: float = 0.20    # solref timeconst -- contact "springiness"
    # SCS0009: 2.3 kg.cm (0.226 N.m) stall at 6 V, and it sags with battery
    # voltage / heat. Scales actuator forcerange (if the scene sets one).
    dr_torque: float = 0.20
    # SCS0009 has a 10-bit magnetic encoder over 300 deg -> 0.293 deg/count
    # = 0.00511 rad. Per-step jitter ~1 count, then quantised to the count grid.
    noise_jpos_rad: float = 0.005
    noise_jvel: float = 0.15
    encoder_quant_rad: float = float(np.radians(300.0) / 1024.0)
    # noise_quat_rad / noise_angvel / noise_linvel: REMOVED. Cube pose/vel
    # noise is no longer injected here -- it is an emergent property of
    # AprilTagEstimator (camera calibration bias, per-detection pose noise,
    # dropout, latency -- ground-truth-based sensor, see apriltag_sim.py).
    # Adding a second, independent noise source on top would double-count
    # and make the sweep in AprilTagCfg unreadable against real logs.
    noise_tip_m: float = 0.006
    action_dropout_p: float = 0.02

    # --- encoder (joint sensor) realism -----------------------------------
    # noise_jpos_rad/noise_jvel above are the per-step measurement noise.
    # These two add what per-step gaussian noise cannot: a persistent
    # miscalibration and a sensor pipeline delay.
    encoder_bias_rad: float = 0.006          # fixed per-episode, per joint (~1 count)
    encoder_latency_steps: tuple = (0, 2)    # inclusive range, drawn per episode

    # --- physical assembly tolerance ---------------------------------------
    # Every finger is bolted on slightly off its CAD zero. Modelled as a
    # persistent per-episode offset added to the TRUE commanded joint angle
    # (not just the sensor reading), so a badly-assembled draw actually
    # changes the achievable grasp geometry, not only what the policy reads.
    joint_assembly_offset_rad: float = 0.01

    # --- SCS0009 command path ---------------------------------------------
    # Servo dead-zone (position hysteresis; SCS default is ~1-2 counts) and
    # gear backlash lump into one per-joint, per-episode deadband drawn from
    # U(0, servo_deadband_rad). Plus a command->motion delay of 0..N control
    # steps (bus write + servo update), drawn per episode.
    servo_deadband_rad: float = 0.010        # ~2 counts max
    action_latency_steps: tuple = (0, 1)

    # --- cube pose source ----------------------------------------------------
    pose_source: str = "gt"                  # "gt" | "apriltag"
    pose: PoseSensorCfg = field(default_factory=PoseSensorCfg)
    # Only used when pose_source == "apriltag".
    apriltag: AprilTagCfg = field(default_factory=AprilTagCfg)


SPIN_CFG = SpinCfg()


# Observation layout, 60 dims. Fingertip-to-cube distance and simulator
# contact state are deliberately excluded -- privileged training-only reward
# signals. tilt_cos/tilt_vec/phase4/angvel/linvel/center_offset are now all
# derived from the pose sensor's delayed, possibly-stale estimate
# (GTPoseSensor by default, AprilTagEstimator if pose_source="apriltag"),
# never raw ground truth. `tag_conf` (fresh_flag, stale_frac) is kept so the
# 60-dim layout -- and therefore exported policies / parity checks -- do not
# change; under "gt" it is (1.0 fresh / 0.0 held, held-run fraction).
SPIN_OBS_SLICES = {
    "jpos": 8, "jvel": 8, "last_action": 8, "prev_action": 8, "filtered_ctrl": 8,
    "tilt_cos": 1, "tilt_vec": 3, "phase4": 2,
    "angvel": 3, "linvel": 3, "center_offset": 3, "axis": 3,
    "tag_conf": 2,
}
SPIN_OBS_DIM = sum(SPIN_OBS_SLICES.values())  # 60


def cube_spin_features(R, k_hat):
    """Pose features for a 4-fold-symmetric object spinning about `k_hat`.

    Absolute yaw about the spin axis is *irrelevant* to this task and, for a
    cube, only meaningful modulo 90 deg. Handing the policy a raw rotation
    matrix would make the observation non-stationary in the one coordinate the
    task is periodic in. Instead:

      tilt_cos : how squarely a cube axis lines up with the spin axis (1.0 =
                 a pair of faces exactly perpendicular to k_hat)
      tilt_vec : which way it is tipped away from that, in the plane normal to
                 k_hat -- the signal for "the cube is about to be lost"
      phase4   : (cos 4psi, sin 4psi) of the roll about k_hat, so the policy can
                 time a push against a face rather than a corner, and the
                 feature repeats every quarter turn exactly as the cube does
    """
    axes = R.T                                   # rows: cube's x, y, z in world
    dots = axes @ k_hat
    i = int(np.argmax(np.abs(dots)))
    a_up = axes[i] * np.sign(dots[i])            # cube axis nearest +k_hat
    tilt_cos = float(np.clip(np.dot(a_up, k_hat), -1.0, 1.0))
    tilt_vec = a_up - tilt_cos * k_hat           # perpendicular component

    # Roll phase: take a cube axis perpendicular to a_up and measure its angle
    # in the plane normal to k_hat.
    j = (i + 1) % 3
    e1 = np.array([k_hat[1], -k_hat[2], k_hat[0]])       # any vector off k_hat
    e1 = e1 - np.dot(e1, k_hat) * k_hat
    n1 = np.linalg.norm(e1)
    e1 = e1 / n1 if n1 > 1e-8 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(k_hat, e1)
    a_p = axes[j] - np.dot(axes[j], k_hat) * k_hat
    psi = float(np.arctan2(np.dot(a_p, e2), np.dot(a_p, e1)))
    return tilt_cos, tilt_vec.astype(np.float32), np.array(
        [np.cos(4 * psi), np.sin(4 * psi)], dtype=np.float32)


class AmazingHandSpinEnv(MujocoEnv):
    """Rotate the cube continuously about one fixed axis, without dropping it."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=CUBE_SCENE, render_mode=None, randomize=True,
                 sensor_noise=True, cfg=SPIN_CFG, **kw):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo scene not found: {model_path}")
        super().__init__(model_path, frame_skip=10, render_mode=render_mode)
        self.cfg = cfg
        self.randomize = randomize
        self.sensor_noise = sensor_noise

        self.actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvelids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        cj = self.model.body("cube").jntadr[0]
        self.cube_qpos = self.model.jnt_qposadr[cj]
        self.cube_dof = self.model.jnt_dofadr[cj]

        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        self.cube_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] == self.cubeid}
        self.finger_bodies = [self._finger_subtree(f) for f in range(N_FINGERS)]
        self.finger_geoms = [{g for g in range(self.model.ngeom)
                              if self.model.geom_bodyid[g] in bodies}
                             for bodies in self.finger_bodies]
        self.finger_body_ids = [np.array(sorted(b)) for b in self.finger_bodies]

        self._base = {
            "geom_friction": self.model.geom_friction.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "dof_frictionloss": self.model.dof_frictionloss.copy(),
            "dof_armature": self.model.dof_armature.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
            "geom_solref": self.model.geom_solref.copy(),
            "actuator_forcerange": self.model.actuator_forcerange.copy(),
        }

        # Cube geometry is READ FROM THE MODEL, not hardcoded. The module
        # constants HALF = 0.0235 and CUBE_LOCAL_CENTER were baked in at the
        # scene's original scale, so scaling the mesh (the only way to resize a
        # mesh geom -- MuJoCo silently ignores `size` on one) left the contact
        # test, the reward and the drop detector all reading the old size. That
        # is what corrupted the first geometry sweep: a cube lying on the floor
        # scored as "held".
        #
        # half-extent: the mesh is stored rotated in its geom frame, so its AABB
        # overstates the cube. The farthest vertex is a corner at HALF*sqrt(3),
        # which is rotation-invariant and therefore safe to invert.
        # The cube body carries seven geoms: the mesh plus six AprilTag decals
        # of size 0.005. Picking an arbitrary one returns a tag and reports the
        # cube as 1 cm across.
        mesh_geoms = [g for g in self.cube_geoms
                      if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        if not mesh_geoms:
            raise RuntimeError("cube body has no mesh geom; cannot infer its size")
        g = mesh_geoms[0]
        mid = self.model.geom_dataid[g]
        if mid >= 0:
            v0 = self.model.mesh_vertadr[mid]
            nv = self.model.mesh_vertnum[mid]
            # `mesh_vert` already has the XML `scale=` attribute baked in at
            # compile time (empirically verified: a unit cube mesh with
            # scale="0.8 0.8 0.8" reads back with vertex norms already x0.8).
            # `model.mesh_scale` is a separate, redundant record of the same
            # factor -- multiplying it back in here double-applies the scale.
            # Invisible at this scene's ~1.0 scale; will silently corrupt
            # cube_half for any --scene-variants scene whose scale actually
            # differs from 1.
            verts = self.model.mesh_vert[v0:v0 + nv]
            self.cube_half = float(np.max(np.linalg.norm(verts, axis=1)) / np.sqrt(3.0))
        else:
            self.cube_half = float(np.max(self.model.geom_size[g]))
        # the mesh is not centred on its body origin; the inertial frame is
        self.cube_local_center = self.model.body_ipos[self.cubeid].copy()

        cube_xyz = self.init_qpos[self.cube_qpos:self.cube_qpos + 3]
        self.nominal_cube_center = cube_xyz.copy() + self.cube_local_center

        lims = self.model.actuator_ctrlrange[self.actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(SPIN_OBS_DIM,),
                                            dtype=np.float32)

        self.k_hat = SPIN_AXES[cfg.spin_axis].copy()
        self.last_action = np.zeros(N_JOINTS, np.float32)
        self.prev_action = np.zeros(N_JOINTS, np.float32)
        self.filtered_ctrl = np.zeros(N_JOINTS, np.float32)
        self.grasp_frac = np.zeros(N_JOINTS, np.float32)
        self.grasp_qpos = np.zeros(N_JOINTS)
        self.start_pos = np.zeros(3)
        self.step_count = 0
        self._ctrl_rate = cfg.max_ctrl_rate_frac
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._rot_acc = 0.0
        self._rot_peak = 0.0
        self._reward_sums = {}
        self._prev_R = np.eye(3)
        self._prev_center = np.zeros(3)
        self._omega_f = np.zeros(3)
        self._finger_ema = np.zeros(N_FINGERS)
        self._reset_tries = 0
        # CURRICULUM KNOB, not a way to skip realism: 1.0 = exactly
        # `cfg`/`cfg.apriltag` (full sim2real spec). Below that, physics
        # jitter spreads, sensor biases and the AprilTag pipeline's own
        # noise/dropout/latency all shrink toward the nominal case
        # continuously -- the tag pipeline is still read every step at every
        # scale, it's just an easier draw from it. Set via `set_dr_scale`;
        # ramp it from the training loop once the policy is doing something
        # at the current scale (see DRCurriculumCallback in train_sac.py).
        self.dr_scale = 1.0

        # Perception: one estimator per env, fed the true cube pose only as
        # "what a camera would have seen" -- see the note by the imports.
        # Nominal tag mount read straight from the compiled model's decal
        # geoms (single source of truth with what's rendered), not guessed
        # or hand-copied -- see AprilTagEstimator's tag_local_poses docstring.
        if cfg.pose_source not in ("gt", "apriltag"):
            raise ValueError(f"pose_source must be 'gt' or 'apriltag', "
                             f"got {cfg.pose_source!r}")
        self._tag_estimator = None
        self._pose_sensor = None
        if cfg.pose_source == "apriltag":
            self._tag_estimator = AprilTagEstimator(
                FACE_NORMALS, cfg.apriltag,
                tag_local_poses=self._read_tag_local_poses())
        else:
            self._pose_sensor = GTPoseSensor(cfg.pose)
        # SCS0009 command path state (see step()).
        self._servo_deadband = np.zeros(N_JOINTS)
        self._servo_applied = np.zeros(N_JOINTS)
        self._cmd_queue = deque(maxlen=cfg.action_latency_steps[1] + 1)
        # Encoder-side realism: persistent per-episode bias + a small
        # latency queue, independent of the AprilTag pipeline's own queue.
        self._encoder_bias = np.zeros(N_JOINTS)
        self._joint_assembly_offset = np.zeros(N_JOINTS)
        self._encoder_queue = deque(maxlen=cfg.encoder_latency_steps[1] + 1)
        self._tag_visible_sum = 0.0
        self._tag_stale_sum = 0.0
        self._ray_geomid = np.zeros(1, dtype=np.int32)
        self._last_occ = set()
        self._occluder_counts = {}      # geom name -> steps it hid a tag (diagnostics)
        # Episode diagnostics (info only, never observed): is the hand actually
        # commanding motion, is it touching the cube, does the estimate track?
        self._diag = self._zero_diag()
        self._reset_valid = True
        self._reset_close_used = cfg.hand_close_frac
        self._eval_resets = 0

        self.close_sign = self._detect_close_sign()

    # ------------------------------------------------------------------
    # Model introspection (shared with the face-target env)
    # ------------------------------------------------------------------
    def _read_tag_local_poses(self):
        """Read tag_number_0..5 decal geom pose (body-local) straight off
        the compiled model, and assign each one to its FACE_NORMALS index
        by GEOMETRY, not by naming convention.

        tag_number_i is NOT FACE_NORMALS[i], and is not any other fixed
        numeric offset either -- two guesses at a formula here were both
        wrong (confirmed against hand.py's actual FACE_NORMALS, which is
        ordered as opposite pairs [+Z,-Z,+Y,-Y,+X,-X], unrelated to any
        1-6 tag-print numbering). The only thing that can't be wrong is
        geometry: which physical face a decal sits on is determined by
        which direction it is from the cube's true centre, so that's what
        this computes -- for each tag_number_i, the direction from
        `body_ipos` (the cube's real geometric centre, read from the model,
        not hardcoded) to the decal's position, matched to the nearest
        FACE_NORMALS entry by dot product. This needs no assumption about
        naming, printing, or spawn orientation, and is correct for any
        scene.xml, including scaled --scene-variants ones.
        """
        n_tags = len(FACE_NORMALS)
        poses = [None] * n_tags
        for i in range(n_tags):
            name = f"tag_number_{i}"
            try:
                g = self.model.geom(name).id
            except KeyError:
                raise RuntimeError(f"scene missing decal geom '{name}'")
            if self.model.geom_bodyid[g] != self.cubeid:
                raise RuntimeError(f"geom '{name}' not attached to cube body")
            pos = self.model.geom_pos[g].copy()
            d = pos - self.cube_local_center
            n = np.linalg.norm(d)
            if n < 1e-6:
                raise RuntimeError(f"geom '{name}' sits at the cube centre; "
                                    "cannot determine which face it's on")
            dots = FACE_NORMALS @ (d / n)
            face_idx = int(np.argmax(dots))
            if dots[face_idx] < 0.5:
                raise RuntimeError(
                    f"geom '{name}' isn't clearly aligned with any face "
                    f"(best dot={dots[face_idx]:.2f} with FACE_NORMALS[{face_idx}]) "
                    "-- check its pos in scene.xml")
            if poses[face_idx] is not None:
                raise RuntimeError(
                    f"both a previous tag and '{name}' matched "
                    f"FACE_NORMALS[{face_idx}] -- scene.xml decal placement "
                    "doesn't cleanly cover all six faces")
            tag_R = quat_to_mat(self.model.geom_quat[g].copy())
            # The estimator does NOT rely on the decal's own z axis (it uses
            # the face normal), but a decal whose z does not point outward is
            # almost certainly a scene.xml mistake and will also render
            # back-to-front, so say so.
            zdot = float(np.dot(tag_R[:, 2], FACE_NORMALS[face_idx]))
            if zdot < 0.9:
                warnings.warn(
                    f"decal '{name}' local +z has dot {zdot:+.2f} with the outward "
                    f"normal of face {face_idx}; check its quat in scene.xml "
                    "(estimator visibility is unaffected, rendering may be)")
            # Centre-relative, NOT body-origin-relative: the env feeds the
            # estimator `_cube_center_world()`, and the mesh centre sits
            # ~3.5 cm from the body origin. `d` is already pos - centre.
            poses[face_idx] = (d.copy(), tag_R)
        if any(p is None for p in poses):
            raise RuntimeError("tag_number_0..5 did not cover all six faces")
        return poses

    @property
    def dt(self):
        """Control-step duration. The base MujocoEnv does not define this, and
        every angular-velocity quantity in this file is a per-control-step
        finite difference, so it has to be right: 10 x 2 ms = 20 ms = 50 Hz."""
        return self.model.opt.timestep * self.frame_skip

    def _finger_subtree(self, finger_idx):
        """All bodies descending from either of finger `finger_idx`'s motors."""
        roots = {self.model.jnt_bodyid[self.model.joint(JOINTS[2 * finger_idx + k]).id]
                 for k in (0, 1)}
        out = set(roots)
        changed = True
        while changed:
            changed = False
            for b in range(self.model.nbody):
                if b not in out and self.model.body_parentid[b] in out:
                    out.add(b)
                    changed = True
        return out

    def _detect_close_sign(self):
        """Which control sign drives the fingers toward the cube."""
        qpos_save = self.data.qpos.copy()
        dists = {}
        for sign in (1.0, -1.0):
            self.data.qpos[self.qposids] = (self.ctrl_mid
                                            + sign * CLOSE_PROBE_FRAC * self.ctrl_half)
            mujoco.mj_forward(self.model, self.data)
            cpos = self._cube_center_world()
            dists[sign] = float(np.mean([np.linalg.norm(self.data.site_xpos[s] - cpos)
                                         for s in self.tip_sites]))
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        return 1.0 if dists[1.0] < dists[-1.0] else -1.0

    # ------------------------------------------------------------------
    # State readout
    # ------------------------------------------------------------------
    def _cube_R(self):
        return self.data.xmat[self.cubeid].reshape(3, 3)

    def _cube_center_world(self):
        return self.data.xpos[self.cubeid] + self._cube_R() @ self.cube_local_center

    # -- public accessors -------------------------------------------------
    # Evaluation, replay and diagnostics all need the cube's orientation and
    # the contact mask. They are exposed here so those tools do not reach into
    # private methods.

    def cube_rotation(self):
        """The cube's current world-frame rotation matrix."""
        return self._cube_R()

    def fingers_touching(self):
        """Boolean mask, one entry per finger, of which are on the cube."""
        return self._fingers_touching_mask()

    def _finger_gaps(self):
        cpos = self._cube_center_world()
        gaps = np.empty(N_FINGERS, dtype=np.float32)
        for f, bids in enumerate(self.finger_body_ids):
            d = np.linalg.norm(self.data.xpos[bids] - cpos[None, :], axis=1)
            gaps[f] = float(np.min(d)) - self.cube_half
        return gaps

    def _tip_to_cube(self):
        cpos = self._cube_center_world()
        return np.array([cpos - self.data.site_xpos[sid] for sid in self.tip_sites],
                        dtype=np.float32)

    def _fingers_touching_mask(self):
        c_force = self.cfg.contact_force_thresh
        touched = np.zeros(N_FINGERS, dtype=bool)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 not in self.cube_geoms and con.geom2 not in self.cube_geoms:
                continue
            other = con.geom2 if con.geom1 in self.cube_geoms else con.geom1
            for f, geoms in enumerate(self.finger_geoms):
                if other in geoms and not touched[f]:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    if abs(f6[0]) > c_force:
                        touched[f] = True
        return touched

    def _touched_faces(self):
        """Which cube faces (indices into FACE_NORMALS) currently have a
        finger pressing on them -- used only to occlude AprilTags, itself a
        privileged shortcut (real occlusion reasoning would be vision-based
        too) but a defensible one: it's a *contact* fact, not a pose fact,
        and it only ever hides information, never reveals it.

        The face is read from the CONTACT NORMAL (which face the finger is
        actually pushing on). The previous rule took the face whose normal best
        matched (fingertip - cube centre); a tip curled over the edge of a side
        face sits above the centre and can label the TOP face pressed, which
        wrongly blinds the one tag the camera sees best. The tip-direction rule
        is kept only as a fallback for a touching finger with no usable contact.
        """
        c_force = self.cfg.contact_force_thresh
        R = self._cube_R()
        faces, resolved = set(), np.zeros(N_FINGERS, dtype=bool)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            cube_first = con.geom1 in self.cube_geoms
            if not cube_first and con.geom2 not in self.cube_geoms:
                continue
            other = con.geom2 if cube_first else con.geom1
            for f, geoms in enumerate(self.finger_geoms):
                if other in geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    if abs(f6[0]) > c_force:
                        # frame[:3] points geom1 -> geom2; make it point out of the cube
                        n_out = np.asarray(con.frame[:3]) * (1.0 if cube_first else -1.0)
                        faces.add(int(np.argmax(np.asarray(FACE_NORMALS) @ (R.T @ n_out))))
                        resolved[f] = True
        touching = self._fingers_touching_mask()
        if (touching & ~resolved).any():
            cpos = self._cube_center_world()
            for f in np.nonzero(touching & ~resolved)[0]:
                local_dir = R.T @ (self.data.site_xpos[self.tip_sites[f]] - cpos)
                faces.add(int(np.argmax(np.asarray(FACE_NORMALS) @ local_dir)))
        return faces

    def _measured_angvel(self):
        """World-frame cube angular velocity, by finite-differencing the pose
        over one *control* step.

        Deliberately not `qvel[3:6]`. That is a body-frame substep value which
        has to be rotated into the world (getting this wrong is what inverted
        the old spin reward once the cube had actually turned), and it is not
        what a deployed system can measure. Differencing the tracked pose over
        the control interval is both correct and reproducible on hardware --
        it is also what Hora does.
        """
        R = self._cube_R()
        return so3_log(R @ self._prev_R.T) / self.dt

    def _cube_center_vel(self):
        """Cube *centre* velocity, finite-differenced over the control step.

        `qvel[cube_dof:cube_dof+3]` is the velocity of the body origin, and the
        cube mesh's centre sits 3.5 cm away from it, so rotating the cube at
        0.5 rad/s registers there as ~1.8 cm/s of translation that is not
        happening. Penalising that would be penalising the task.
        """
        return (self._cube_center_world() - self._prev_center) / self.dt

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------
    def _apply_domain_randomization(self):
        c, rng = self.cfg, self.np_random
        b = self._base
        self.model.geom_friction[:] = b["geom_friction"]
        self.model.dof_damping[:] = b["dof_damping"]
        self.model.dof_frictionloss[:] = b["dof_frictionloss"]
        self.model.dof_armature[:] = b["dof_armature"]
        self.model.body_mass[:] = b["body_mass"]
        self.model.body_inertia[:] = b["body_inertia"]
        self.model.actuator_gainprm[:] = b["actuator_gainprm"]
        self.model.actuator_biasprm[:] = b["actuator_biasprm"]
        self.model.geom_solref[:] = b["geom_solref"]
        self.model.actuator_forcerange[:] = b["actuator_forcerange"]
        self._ctrl_rate = c.max_ctrl_rate_frac
        self._servo_deadband[:] = 0.0
        self._encoder_bias[:] = 0.0
        self._joint_assembly_offset[:] = 0.0
        if not self.randomize:
            mujoco.mj_setConst(self.model, self.data)
            return

        def jit(spread, size=None):
            return rng.uniform(1.0 - spread * s, 1.0 + spread * s, size)

        s = self.dr_scale
        self.model.geom_friction[:, 0] *= jit(c.dr_friction, self.model.ngeom)
        self.model.geom_friction[:, 1] *= jit(c.dr_friction, self.model.ngeom)
        self.model.geom_friction[:, 2] *= jit(c.dr_rolling_friction, self.model.ngeom)
        self.model.dof_damping[:] *= jit(c.dr_damping, self.model.nv)
        self.model.dof_frictionloss[:] *= jit(c.dr_damping, self.model.nv)
        self.model.dof_armature[:] *= jit(c.dr_damping, self.model.nv)
        m = float(jit(c.dr_mass))
        self.model.body_mass[self.cubeid] *= m
        self.model.body_inertia[self.cubeid] *= m
        # Hand/finger link mass+inertia: a real hand's links don't match CAD
        # nominal mass either (fasteners, wiring, print variance). Every body
        # except the cube (jittered above, its own spread) and world (id 0,
        # mass 0 by convention -- scaling it is a no-op but keep the
        # exclusion explicit).
        other_bodies = np.array([b for b in range(1, self.model.nbody)
                                  if b != self.cubeid])
        if len(other_bodies):
            lm = jit(c.dr_link_mass, len(other_bodies))
            self.model.body_mass[other_bodies] *= lm
            self.model.body_inertia[other_bodies] *= lm[:, None]
        # Contact softness: solref[0] is the effective time-constant of the
        # contact spring-damper. Real fingertip/cube material compliance
        # varies episode to episode (wear, temperature, print batch); a
        # policy trained on one fixed softness overfits to it.
        self.model.geom_solref[:, 0] *= jit(c.dr_contact_softness, self.model.ngeom)
        # A position actuator's stiffness lives in two places; changing only
        # gainprm leaves the bias term inconsistent and silently detunes the
        # servo instead of stiffening it.
        g = jit(c.dr_gain, len(self.actids))
        self.model.actuator_gainprm[self.actids, 0] *= g
        self.model.actuator_biasprm[self.actids, 1] *= g
        self._ctrl_rate = c.max_ctrl_rate_frac * float(jit(c.dr_ctrl_rate))
        # Torque ceiling (voltage sag / thermal derating), per actuator, only
        # where the scene actually limits force.
        if np.any(self.model.actuator_forcelimited[self.actids]):
            fr = jit(c.dr_torque, len(self.actids))
            self.model.actuator_forcerange[self.actids] *= fr[:, None]
        self._servo_deadband[:] = rng.uniform(0.0, c.servo_deadband_rad * s, N_JOINTS)
        # Sensor-side calibration error (what the policy sees is wrong) and
        # physical assembly error (what the hand actually does is wrong) are
        # drawn independently -- a well-calibrated encoder on a badly-bolted
        # finger, and vice versa, are both real failure modes.
        self._encoder_bias[:] = rng.normal(0.0, c.encoder_bias_rad * s, N_JOINTS)
        self._joint_assembly_offset[:] = rng.normal(0.0, c.joint_assembly_offset_rad * s,
                                                     N_JOINTS)
        mujoco.mj_setConst(self.model, self.data)

    def set_dr_scale(self, scale: float):
        """Curriculum entry point: 0 = nominal-strength physics/sensor jitter
        with the AprilTag pipeline still fully in the loop (just an easier
        draw from it -- see AprilTagEstimator.reset); 1 = exactly `cfg`, the
        real, fully-randomised spec. Callable via VecEnv.env_method so a
        training-loop callback can ramp every env at once."""
        self.dr_scale = float(np.clip(scale, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._reset_options = options or {}
        return super().reset(seed=seed, options=options)

    @staticmethod
    def _zero_diag():
        return {"act_abs": 0.0, "ctrl_travel": 0.0, "touch": 0.0, "touch2": 0.0,
                "track_err": 0.0, "est_rot": 0.0, "ray_occ": 0.0, "boot": 0.0, "n": 0}

    def _attempt_score(self, n_touch, drift):
        c = self.cfg
        return min(n_touch, c.min_start_contacts) * 10.0 - drift / c.max_start_drift_m

    def reset_model(self):
        """Place the cube, close the hand, and *verify* the grasp took.

        Hora keeps a cache of pre-validated grasps and samples from it. The
        equivalent here is to re-roll the settle until it produces a real grasp:
        at least `min_start_contacts` fingers bearing force and the cube still
        near where it was placed.

        If NO attempt passes, the previous version fell out of the loop and
        started the episode from the last (arbitrary) attempt. Now the closure
        is adjusted between attempts, the best attempt is re-run and kept, and
        `_reset_valid` is recorded so every episode says whether it started
        from a real grasp (`ep_reset_valid`). `cfg.strict_reset` raises instead.
        """
        c, rng = self.cfg, self.np_random
        opts = getattr(self, "_reset_options", None) or {}

        if c.randomize_axis_sign and "spin_axis" not in opts:
            self.k_hat = SPIN_AXES[c.spin_axis].copy() * float(rng.choice([-1.0, 1.0]))
        elif "spin_axis" in opts:
            self.k_hat = SPIN_AXES[opts["spin_axis"]].copy()
        else:
            self.k_hat = SPIN_AXES[c.spin_axis].copy()

        # Physics randomisation is drawn ONCE per episode, before the attempts.
        # It used to be redrawn inside every retry, so each attempt was judged
        # under different friction/gains and a "best" attempt could not be
        # reproduced. mj_resetData inside the attempt does not touch the model.
        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_randomization()

        # start face for deterministic (eval) resets: cycle, so N episodes cover
        # the faces evenly without touching the RNG.
        if not self.randomize:
            self._eval_face = self._eval_resets % len(FACE_NORMALS)
            self._eval_resets += 1

        close = float(c.hand_close_frac)          # closure for the NEXT attempt
        live_close = close                        # closure of the attempt now in `data`
        best = None                               # (score, attempt, snapshot, closure)
        valid, n_touch, drift = False, 0, float("inf")
        attempts = max(int(c.max_reset_tries), 1)
        for attempt in range(attempts):
            live_close = close
            frac_cmd = self._place_and_settle(rng, opts, live_close)
            n_touch = int(self._fingers_touching_mask().sum())
            drift = float(np.linalg.norm(self._cube_center_world() - self._placed_center))
            valid = (n_touch >= c.min_start_contacts and drift <= c.max_start_drift_m)
            score = self._attempt_score(n_touch, drift)
            if best is None or score > best[0]:
                # Snapshot the committed sim state directly rather than the RNG
                # state -- replaying the settle via a rewound RNG discards and
                # re-consumes every random draw made by attempts in between,
                # and re-simulates SETTLE_CTRL_STEPS for nothing. A snapshot
                # is exact, free to restore, and touches the RNG not at all.
                best = (score, attempt,
                        {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                         "ctrl": self.data.ctrl.copy(),
                         "placed_center": self._placed_center.copy(),
                         "frac_cmd": frac_cmd.copy()},
                        live_close)
            if valid:
                break
            # Adjust the closure for the next attempt according to WHY it failed.
            if n_touch < c.min_start_contacts:
                close = min(close + c.reset_close_step, 1.0)
            elif drift > c.max_start_drift_m:
                close = max(close - c.reset_close_step, 0.0)
        self._reset_tries = attempt + 1

        if not valid:
            if c.strict_reset:
                raise RuntimeError(
                    f"no valid grasp after {attempts} attempts (best score attempt "
                    f"{best[1] + 1}; last: {n_touch} fingers touching, drift "
                    f"{drift * 1000:.1f} mm; need >= {c.min_start_contacts} and "
                    f"<= {c.max_start_drift_m * 1000:.0f} mm). Try a different "
                    f"--close-frac or check the scene geometry.")
            if best[1] != attempt:
                # Restore the BEST attempt's committed state instead of the
                # last one. mj_forward recomputes everything derived
                # (xpos/xmat/contacts) from the restored qpos/qvel -- no
                # resimulation, no RNG involved.
                snap = best[2]
                self.data.qpos[:] = snap["qpos"]
                self.data.qvel[:] = snap["qvel"]
                self.data.ctrl[:] = snap["ctrl"]
                mujoco.mj_forward(self.model, self.data)
                self._placed_center = snap["placed_center"]
                frac_cmd = snap["frac_cmd"]
                live_close = best[3]
        close = live_close
        self._reset_valid = bool(valid)
        self._reset_close_used = float(close)

        frac_now = np.clip(
            (self.data.qpos[self.qposids] - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        if c.anchor_grasp_on_command:
            # Continuous with the settle: keep commanding what the servos were
            # already being commanded, so the squeeze the cube is held with
            # does not disappear on step 1.
            frac_init = frac_cmd
        else:
            frac_init = frac_now
        self.grasp_frac[:] = frac_init
        self.filtered_ctrl[:] = frac_init
        self.grasp_qpos[:] = self.data.qpos[self.qposids]
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.start_pos = self._cube_center_world().copy()
        self._prev_R = self._cube_R().copy()
        self._prev_center = self.start_pos.copy()
        self._omega_f = np.zeros(3)
        # start at 1.0 so an episode is not charged for idleness it has not had
        # time to demonstrate
        self._finger_ema[:] = 1.0

        self.step_count = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._rot_acc = 0.0
        self._rot_peak = 0.0
        self._reward_sums = {}
        self._tag_visible_sum = 0.0
        self._tag_stale_sum = 0.0
        self._diag = self._zero_diag()

        # Perception/encoder pipelines are re-drawn every episode, same as
        # every other domain-randomised quantity. When sensor_noise=False
        # (eval) the estimator is nominal: no noise, latency or dropout -- but
        # it still only sees geometrically visible, unoccluded tags.
        if self._tag_estimator is not None:
            self._tag_estimator.reset(rng, self.cube_half, nominal=not self.sensor_noise,
                                      cam_target=self.nominal_cube_center,
                                      dr_scale=self.dr_scale)
        else:
            self._pose_sensor.reset(rng, self._cube_R(), self._cube_center_world(),
                                    nominal=not self.sensor_noise,
                                    dr_scale=self.dr_scale)
        # SCS0009 command path: start already at the commanded grasp so step 1
        # sees no phantom deadband jump or stale-queue command.
        cmd0 = self.ctrl_mid + self.filtered_ctrl * self.ctrl_half
        self._servo_applied[:] = cmd0
        cl_lo = int(round(c.action_latency_steps[0] * self.dr_scale))
        cl_hi = max(cl_lo, int(round(c.action_latency_steps[1] * self.dr_scale)))
        n_cmd = int(rng.integers(cl_lo, cl_hi + 1)) if self.sensor_noise else 0
        self._cmd_queue = deque([cmd0.copy()] * (n_cmd + 1), maxlen=n_cmd + 1)
        lat_lo = int(round(c.encoder_latency_steps[0] * self.dr_scale))
        lat_hi = max(lat_lo, int(round(c.encoder_latency_steps[1] * self.dr_scale)))
        n_lat = int(rng.integers(lat_lo, lat_hi + 1)) if self.sensor_noise else 0
        self._encoder_queue = deque(maxlen=n_lat + 1)
        self._refresh_tag_estimate()

    def _place_and_settle(self, rng, opts, close_frac):
        """One placement + settle. Returns the closure COMMAND (as an action
        fraction, without the per-episode assembly offset) the servos were
        being driven to at the end of the settle.

        Physics randomisation is NOT applied here (see `reset_model`).
        """
        c = self.cfg
        # Retries have to start from the same clean state, or attempt 2
        # inherits attempt 1's passive linkage angles.
        mujoco.mj_resetData(self.model, self.data)

        qpos_open = np.clip(
            self.ctrl_mid + self.close_sign * GRASP_OPEN_FRAC * self.ctrl_half,
            self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qposids] = qpos_open
        self.data.qvel[self.qvelids] = 0.0
        self.data.ctrl[self.actids] = qpos_open

        # Training resets are randomised; evaluation resets (randomize=False)
        # are not, except that the face cycles. A `start_face` option wins.
        jitter = self.randomize
        base_quat = self.init_qpos[self.cube_qpos + 3:self.cube_qpos + 7].copy()
        if "start_face" in opts:
            face = int(opts["start_face"])
        elif jitter:
            face = int(rng.integers(6)) if c.randomize_start_face else 0
        else:
            face = int(getattr(self, "_eval_face", 0))
        # FACE_NORMALS are cube-BODY directions; put the chosen one on +Z given
        # whatever the cube's initial orientation is (identity in the stock
        # scene, in which case this is exactly the previous behaviour).
        n_world = quat_to_mat(base_quat) @ FACE_NORMALS[face]
        quat = qmul(q_from_vecs(n_world, np.array([0.0, 0.0, 1.0])), base_quat)
        if jitter:
            # Full 180 deg of roll jitter: for a continuous-rotation task the
            # phase at which the episode starts carries no information, and
            # pinning it would let the policy memorise one entry point.
            quat = qmul(q_from_axis_angle([0, 0, 1],
                                          float(rng.uniform(-c.yaw_jitter_rad,
                                                            c.yaw_jitter_rad))), quat)
            tilt_axis = rng.normal(size=3)
            tilt_axis[2] = 0.0
            quat = qmul(q_from_axis_angle(tilt_axis + np.array([1e-6, 0, 0]),
                                          float(rng.uniform(-c.tilt_jitter_rad,
                                                            c.tilt_jitter_rad))), quat)
        quat /= np.linalg.norm(quat)

        target_center = self.nominal_cube_center.copy()
        if jitter:
            target_center[:2] += rng.uniform(-c.pos_jitter_m, c.pos_jitter_m, 2)
        self.data.qpos[self.cube_qpos:self.cube_qpos + 3] = (
            target_center - quat_to_mat(quat) @ self.cube_local_center)
        self.data.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = quat
        self.data.qvel[self.cube_dof:self.cube_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._placed_center = self._cube_center_world().copy()

        qpos_closed = self.ctrl_mid + self.close_sign * close_frac * self.ctrl_half
        if jitter:
            qpos_closed = qpos_closed + rng.uniform(-HAND_CLOSE_JITTER,
                                                    HAND_CLOSE_JITTER, N_JOINTS)
        qpos_closed = np.clip(qpos_closed, self.ctrl_lo, self.ctrl_hi)
        full = np.zeros(self.model.nu)
        for i in range(SETTLE_CTRL_STEPS):
            frac = (i + 1) / SETTLE_CTRL_STEPS
            full[self.actids] = (qpos_open + frac * (qpos_closed - qpos_open)
                                 + self._joint_assembly_offset)
            self.do_simulation(full, self.frame_skip)
        return np.clip((qpos_closed - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _refresh_tag_estimate(self):
        """Call exactly once per control step (reset and step, never more --
        the estimator's latency queue and staleness counter are stateful and
        would desync from double-calls). Stores the result for `_get_obs`."""
        if self._tag_estimator is None:
            self._last_occ = set()
            self._tag_est = self._pose_sensor.step(
                self._cube_R(), self._cube_center_world(), self.dt)
            return
        self._last_occ = self._ray_occluded_faces()
        self._tag_est = self._tag_estimator.step(
            self._cube_center_world(), self._cube_R(),
            touched_faces=self._touched_faces(), dt=self.dt,
            occluded_faces=self._last_occ)

    def _ray_occluded_faces(self):
        """Faces whose tag has a NON-cube geom between it and the camera.

        A tag is hidden by a finger that sits in front of it, not only by a
        finger pressing that very face. One ray per tag from the (true) camera
        position to the tag centre; the first thing hit must be part of the cube
        (its mesh or a decal). Any hand link, the wrist, or the base in front of
        it occludes. Privileged (uses MuJoCo geometry) but it only ever HIDES
        information, like `_touched_faces`.

        CHECK `diagnose_spin` for the occluder names: a geom that hides tags
        but is not physically in the camera's way (an invisible helper geom, a
        marker) is a scene bug, not real occlusion."""
        c = self.cfg.apriltag
        if not c.raycast_occlusion:
            return set()
        est = self._tag_estimator
        origin = np.asarray(est.cam_pos, dtype=np.float64)
        occ = set()
        for i, tp in enumerate(est.tag_positions_world(self._cube_center_world(),
                                                       self._cube_R())):
            vec = np.asarray(tp, dtype=np.float64) - origin
            d = float(np.linalg.norm(vec))
            if d < 1e-6:
                continue
            dist = mujoco.mj_ray(self.model, self.data, origin, vec / d, None, 1, -1,
                                 self._ray_geomid)
            gid = int(self._ray_geomid[0])
            if dist >= 0.0 and gid >= 0 and gid not in self.cube_geoms \
                    and dist < d - c.ray_tol_m:
                occ.add(i)
                name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                        or "geom%d@%s" % (gid, self.model.body(
                            int(self.model.geom_bodyid[gid])).name))
                self._occluder_counts[name] = self._occluder_counts.get(name, 0) + 1
        return occ

    def _encoder_reading(self):
        """Joint sensors: per-step gaussian noise (existing) + a persistent
        per-episode calibration bias + a small pipeline latency queue,
        independent of the AprilTag pipeline's own queue."""
        c, rng = self.cfg, self.np_random
        jpos_raw = self.data.qpos[self.qposids].copy()
        jvel_raw = self.data.qvel[self.qvelids].copy()
        if self.sensor_noise:
            jpos_raw = (jpos_raw + self._encoder_bias
                       + rng.normal(0.0, c.noise_jpos_rad * self.dr_scale, N_JOINTS))
            jvel_raw = jvel_raw + rng.normal(0.0, c.noise_jvel * self.dr_scale, N_JOINTS)
            q = c.encoder_quant_rad
            jpos_raw = np.round(jpos_raw / q) * q       # 10-bit SCS0009 count grid
        self._encoder_queue.append((jpos_raw, jvel_raw))
        return self._encoder_queue[0]

    def _get_obs(self):
        jpos_raw, jvel_raw = self._encoder_reading()
        jpos = np.clip((jpos_raw - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        jvel = np.clip(jvel_raw / JVEL_SCALE, -1.0, 1.0)

        # Everything below comes from AprilTagEstimator's delayed, fused,
        # possibly-held pose estimate -- see `_refresh_tag_estimate`. No
        # ground truth (`_cube_R`/`_cube_center_world`) is read in this
        # method, by construction.
        est = self._tag_est
        tilt_cos, tilt_vec, phase4 = cube_spin_features(est["R"], self.k_hat)

        angvel = np.clip(est["angvel"] / ANGVEL_OBS_SCALE, -3.0, 3.0)
        linvel = np.clip(est["linvel"] / LINVEL_OBS_SCALE, -3.0, 3.0)
        offset = np.clip((est["pos"] - self.nominal_cube_center) / 0.05, -3.0, 3.0)
        tag_conf = np.array([est["n_visible"] / 6.0, est["stale_frac"]],
                            dtype=np.float32)

        # Privileged signals are intentionally NOT exposed to the actor.
        return np.concatenate([
            jpos, jvel, self.last_action, self.prev_action, self.filtered_ctrl,
            [tilt_cos], tilt_vec, phase4,
            angvel, linvel, offset, self.k_hat, tag_conf,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward(self, act, dropped):
        """r = w_rot * (omega . k_hat) * dt  -  (penalties, all <= 0).

        The rotation term is an increment, so the episode sum is *exactly*
        w_rot x (net radians turned about k_hat): one revolution = +50. Signed,
        so turning one way and back nets zero -- it pays for progress, not for
        motion -- and there is no clipping artefact to farm.

        The invariant that separates this from every previous version of this
        project's reward: nothing except the rotation term is ever positive, so
        a policy that does nothing scores ~0 and any net rotation beats it.
        Verified numerically by measuring every term over do-nothing,
        random and CEM-rotator rollouts -- see docs/in-hand-rotation.md --
        rather than asserted.
        """
        c = self.cfg
        omega_raw = self._measured_angvel()
        omega = self._omega_f                      # jitter removed; see angvel_lpf
        proj = float(np.clip(np.dot(omega, self.k_hat), -c.angvel_clip, c.angvel_clip))

        r_rot = c.w_rot * proj * self.dt

        offaxis = float(np.linalg.norm(omega - np.dot(omega, self.k_hat) * self.k_hat))
        r_offaxis = -c.w_offaxis * min(offaxis, c.offaxis_clip)

        vel = self._cube_center_vel()
        r_linvel = -c.w_linvel * float(np.sum(np.abs(vel)))

        drift = float(np.linalg.norm(self._cube_center_world() - self.start_pos))
        r_drift = -c.w_drift * drift

        if c.w_pose > 0.0:
            dq = self.data.qpos[self.qposids] - self.grasp_qpos
            r_pose = -c.w_pose * float(np.sum(dq * dq))
        else:
            r_pose = 0.0

        tau = self.data.actuator_force[self.actids]
        qd = self.data.qvel[self.qvelids]
        r_work = -c.w_work * float(np.sum(tau * qd)) ** 2
        r_torque = -c.w_torque * float(np.sum(tau * tau))

        d_act = act - self.last_action
        r_action = -c.w_action_rate * float(np.sum(d_act * d_act))

        # PRIVILEGED TRAINING-ONLY SIGNALS: MuJoCo contact + fingertip distance.
        touching = self._fingers_touching_mask()
        n_touch = int(touching.sum())
        self._diag["touch"] += n_touch
        self._diag["touch2"] += float(n_touch >= 2)
        r_contact = (-c.w_contact_loss
                     * max(0, c.contact_min_fingers - n_touch)
                     / max(c.contact_min_fingers, 1)
                     if c.w_contact_loss > 0.0 else 0.0)

        tip_vecs = self._tip_to_cube()
        tip_dist = np.linalg.norm(tip_vecs, axis=1)
        tip_safe = float(self.cube_half + c.tip_safe_extra_m)
        tip_excess = np.clip(
            tip_dist - tip_safe, 0.0, c.tip_distance_clip_m)
        r_tip_distance = (-c.w_tip_distance
                          * float(np.mean(
                              (tip_excess / max(REACH_NORM, 1e-9)) ** 2))
                          if c.w_tip_distance > 0.0 else 0.0)

        self._finger_ema += c.finger_ema * (touching.astype(float) - self._finger_ema)
        idle = 1.0 - float(self._finger_ema.min())
        r_finger = -c.w_finger_idle * idle if c.w_finger_idle > 0.0 else 0.0

        r_drop = -c.drop_penalty if dropped else 0.0

        total = (r_rot + r_offaxis + r_linvel + r_drift + r_pose + r_work
                 + r_torque + r_action + r_contact + r_finger
                 + r_tip_distance + r_drop)

        # The metric, deliberately computed from the RAW measurement: "the cube
        # rotated N degrees" has to mean the cube actually rotated N degrees,
        # not that a filtered estimate of it did.
        self._rot_acc += float(np.dot(omega_raw, self.k_hat)) * self.dt
        self._rot_peak = max(self._rot_peak, abs(self._rot_acc))

        info = {
            "r_rot": r_rot, "r_offaxis": r_offaxis, "r_linvel": r_linvel,
            "r_drift": r_drift, "r_pose": r_pose, "r_work": r_work,
            "r_torque": r_torque, "r_action": r_action, "r_contact": r_contact,
            "r_finger": r_finger, "r_drop": r_drop,
            "min_finger_ema": float(self._finger_ema.min()),
            "spin_rate": proj, "offaxis_rate": offaxis, "n_touch": n_touch,
            "rot_acc_rad": self._rot_acc, "drift_m": drift,
            "tag_n_visible": self._tag_est["n_visible"],
            "tag_stale_frac": self._tag_est["stale_frac"],
        }
        self._tag_visible_sum += self._tag_est["n_visible"]
        self._tag_stale_sum += self._tag_est["stale_frac"]
        for k, v in info.items():
            if k.startswith("r_"):
                self._reward_sums[k] = self._reward_sums.get(k, 0.0) + float(v)
        return float(total), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        c = self.cfg
        act = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0).astype(np.float32)
        if self.sensor_noise and self.np_random.random() < c.action_dropout_p:
            act = self.last_action.copy()

        target_frac = np.clip(self.grasp_frac + act * c.grasp_band_frac, -1.0, 1.0)
        lpf_ctrl = c.action_lpf * target_frac + (1 - c.action_lpf) * self.filtered_ctrl
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl,
                             -self._ctrl_rate, self._ctrl_rate)
        self.filtered_ctrl = self.filtered_ctrl + step_delta

        self._prev_R = self._cube_R().copy()
        self._prev_center = self._cube_center_world().copy()
        # SCS0009 command path: bus/servo delay, then dead-zone hysteresis (the
        # servo only moves once the command leaves +-deadband of where it sits).
        self._cmd_queue.append(self.ctrl_mid + self.filtered_ctrl * self.ctrl_half)
        cmd = self._cmd_queue[0]
        err = cmd - self._servo_applied
        move = np.where(np.abs(err) > self._servo_deadband,
                        err - np.sign(err) * self._servo_deadband, 0.0)
        self._servo_applied = self._servo_applied + move
        full = np.zeros(self.model.nu)
        full[self.actids] = self._servo_applied + self._joint_assembly_offset
        self.do_simulation(full, self.frame_skip)
        self.step_count += 1

        cpos = self._cube_center_world()
        z_drop = float(self.start_pos[2] - cpos[2])
        xy_drift = float(np.linalg.norm(cpos[:2] - self.start_pos[:2]))
        self._xy_over_steps = self._xy_over_steps + 1 if xy_drift > c.xy_drop_m else 0
        self._z_over_steps = self._z_over_steps + 1 if z_drop > c.z_drop_m else 0
        dropped = (self._xy_over_steps >= c.drop_persist_steps
                   or self._z_over_steps >= c.drop_persist_steps)

        self._omega_f = ((1.0 - c.angvel_lpf) * self._omega_f
                         + c.angvel_lpf * self._measured_angvel())
        self._refresh_tag_estimate()
        d = self._diag
        d["n"] += 1
        d["act_abs"] += float(np.mean(np.abs(act)))
        d["ctrl_travel"] += float(np.sum(np.abs(step_delta)))
        est = self._tag_est
        d["track_err"] += float(np.linalg.norm(so3_log(est["R"] @ self._cube_R().T)))
        d["est_rot"] += float(np.dot(est["angvel"], self.k_hat)) * self.dt
        d["ray_occ"] += len(self._last_occ)
        d["boot"] += float(est["bootstrap"])
        reward, info = self._reward(act, dropped)
        self.prev_action = self.last_action.copy()
        self.last_action = act.copy()

        terminated = bool(dropped and c.terminate_on_drop)
        truncated = self.step_count >= c.max_steps

        if terminated or truncated:
            revs = self._rot_acc / (2 * np.pi)
            info.update({f"ep_{k}": v for k, v in self._reward_sums.items()})
            info.update({
                "ep_rotation_rad": self._rot_acc,
                "ep_revolutions": revs,
                "ep_rot_per_sec": self._rot_acc / max(self.step_count * self.dt, 1e-9),
                "ep_steps": self.step_count,
                "ep_dropped": float(dropped),
                "ep_reset_tries": self._reset_tries,
                # A ladder of bars, so the curve is readable long before the
                # first full turn. `success` is the headline number.
                "success": float(revs >= c.success_revolutions),
                "reached_quarter_turn": float(revs >= 0.25),
                "reached_half_turn": float(revs >= 0.5),
                "reached_full_turn": float(revs >= 1.0),
                "reached_two_turns": float(revs >= 2.0),
                "ep_tag_visible_mean": self._tag_visible_sum / max(self.step_count, 1),
                "ep_tag_stale_mean": self._tag_stale_sum / max(self.step_count, 1),
                # --- is the hand trying, and does the estimate see it? -------
                # ep_reset_valid     1.0 = the episode began from a verified grasp
                # ep_act_abs_mean    mean |raw policy action| (0 = commanding nothing)
                # ep_ctrl_travel     total distance the filtered command moved (frac
                #                    units); ~0 with high act_abs = slew/LPF eats it
                # ep_touch_mean      mean fingers in contact; ep_touch2_frac >=2 fingers
                # ep_track_err_deg   mean angle between the tag estimate and the truth
                # ep_est_rotation_rad rotation the POLICY perceived about k_hat; compare
                #                    with ep_rotation_rad -- if they disagree the
                #                    observation, not the policy, is the problem
                "ep_reset_valid": float(self._reset_valid),
                "ep_reset_close": float(self._reset_close_used),
                "ep_act_abs_mean": d["act_abs"] / max(d["n"], 1),
                "ep_ctrl_travel": d["ctrl_travel"],
                "ep_touch_mean": d["touch"] / max(d["n"], 1),
                "ep_touch2_frac": d["touch2"] / max(d["n"], 1),
                "ep_track_err_deg": float(np.degrees(d["track_err"] / max(d["n"], 1))),
                "ep_est_rotation_rad": d["est_rot"],
                # tags hidden by geometry between camera and tag (per step), and the
                # fraction of steps the policy had NEVER yet seen a tag (prior only)
                "ep_tag_ray_occ_mean": d["ray_occ"] / max(d["n"], 1),
                "ep_tag_boot_frac": d["boot"] / max(d["n"], 1),
            })

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info


def make_spin_cfg(**overrides):
    return replace(SPIN_CFG, **overrides)


def make_eval_env(**kw):
    """Deterministic evaluation: no domain randomisation, no sensor noise."""
    kw.setdefault("randomize", True)
    kw.setdefault("sensor_noise", True)
    return AmazingHandSpinEnv(**kw)