"""Evaluate a rotation policy, in the units the task is defined in.

Examples
--------
    python -m sohand.rl.evaluate --model runs/spin/models/best_model.zip --show
    python -m sohand.rl.evaluate --model runs/spin/models/best_model.zip --show --episodes 24
    python -m sohand.rl.evaluate --model runs/spin/models/best_model.zip
    python -m sohand.rl.evaluate --model runs/spin/models/best_model.zip --video spin.mp4

Two rotation numbers are reported and they are not the same quantity:

  credited
      the integral of omega . k_hat that the reward pays for.

  swing-twist
      the twist angle of the cube's orientation about k_hat.

The ratio is an honesty check on the headline figure.

Nominal physics with clean sensors measures skill;
--randomize measures robustness.
"""

import argparse
import time

import numpy as np

from sohand import paths
from sohand.envs import AmazingHandSpinEnv, SPIN_CFG, make_spin_cfg
from sohand.envs.spin import SPIN_AXES
from sohand.rl.actor import NumpyActor, RUNS, SinusoidGait
from sohand.rotations import twist_and_swing, unwrap_delta


# Beyond this swing angle the twist about a fixed axis becomes
# too ill-conditioned to accumulate reliably.
MAX_TRACKABLE_SWING = np.radians(150.0)


def rollout(env, policy, axis, seed, frames=None, show=False):
    """Run one episode.

    Returns:
        return,
        info,
        net twist in radians,
        maximum swing angle
    """

    obs, _ = env.reset(seed=seed)

    prev_yaw, swing = twist_and_swing(
        env.cube_rotation(),
        axis,
    )

    net_twist = 0.0
    episode_return = 0.0
    step = 0
    worst_swing = swing

    done = False
    info = {}

    while not done:

        # Get action from trained policy.
        action = policy(obs, step)

        # Step MuJoCo environment.
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward

        # ---------------------------------------------------------
        # LIVE M U J O C O VIEWER
        # ---------------------------------------------------------
        if show:
            env.render()

            # Small delay so the viewer follows approximately
            # real-time physics instead of running at maximum speed.
            if hasattr(env, "dt"):
                time.sleep(max(0.0, float(env.dt)))

        # ---------------------------------------------------------
        # VIDEO FRAME CAPTURE
        # ---------------------------------------------------------
        if frames is not None:
            image = env.render()

            if image is not None:
                frames.append(image)

        # ---------------------------------------------------------
        # ROTATION METRICS
        # ---------------------------------------------------------
        yaw, swing = twist_and_swing(
            env.cube_rotation(),
            axis,
        )

        net_twist += unwrap_delta(
            yaw,
            prev_yaw,
        )

        prev_yaw = yaw

        worst_swing = max(
            worst_swing,
            swing,
        )

        done = terminated or truncated
        step += 1

    return (
        episode_return,
        info,
        net_twist,
        worst_swing,
    )


def build(args):
    """Build environment and policy."""

    scene = args.scene
    close = args.close_frac

    # -------------------------------------------------------------
    # LOAD RELEASED RUN CONFIG
    # -------------------------------------------------------------
    if args.run is not None:

        _, run_scene, run_close = RUNS[args.run]

        if scene is None and run_scene:
            scene = f"{paths.MJCF_DIR}/cube/{run_scene}"

        if close is None:
            close = run_close

    # -------------------------------------------------------------
    # RENDER MODE
    # -------------------------------------------------------------
    #
    # --show  -> actual MuJoCo interactive viewer
    # --video -> RGB frames for video file
    # neither -> no rendering
    #
    if args.show:
        render_mode = "human"

    elif args.video:
        render_mode = "rgb_array"

    else:
        render_mode = None

    # -------------------------------------------------------------
    # ENVIRONMENT
    # -------------------------------------------------------------
    env_kwargs = dict(
        randomize=args.randomize,
        sensor_noise=args.randomize,
        render_mode=render_mode,
    )

    if scene:
        env_kwargs["model_path"] = scene

    if close is not None:
        env_kwargs["cfg"] = make_spin_cfg(
            hand_close_frac=close
        )

    env = AmazingHandSpinEnv(
        **env_kwargs
    )

    # -------------------------------------------------------------
    # POLICY
    # -------------------------------------------------------------

    # Open-loop CEM gait.
    if args.gait:

        gait_file = (
            args.gait_file
            or paths.require_checkpoint("gait_cem.npy")
        )

        policy = SinusoidGait.load(
            gait_file,
            env.dt,
        )

        return (
            env,
            policy,
            "open-loop CEM gait",
        )

    # Released Numpy actor.
    if args.run is not None:

        actor, _, _ = NumpyActor.for_run(
            args.run
        )

        return (
            env,
            actor,
            f"run {args.run} ({RUNS[args.run][0]})",
        )

    # Exported actor.
    if args.actor:

        actor = NumpyActor(
            args.actor
        )

        return (
            env,
            actor,
            args.actor,
        )

    # -------------------------------------------------------------
    # STABLE-BASELINES3 SAC
    # -------------------------------------------------------------
    if args.model:

        from stable_baselines3 import SAC

        model = SAC.load(
            args.model,
            device="cpu",
        )

        def policy(obs, k):
            action, _ = model.predict(
                obs,
                deterministic=not args.stochastic,
            )

            return action

        return (
            env,
            policy,
            args.model,
        )

    # -------------------------------------------------------------
    # DO NOTHING
    # -------------------------------------------------------------
    return (
        env,
        lambda obs, k: np.zeros(
            8,
            dtype=np.float32,
        ),
        "do-nothing",
    )


def main(args):

    # -------------------------------------------------------------
    # BUILD
    # -------------------------------------------------------------
    env, policy, label = build(args)

    axis = SPIN_AXES[
        env.cfg.spin_axis
    ]

    # Only store frames when making a video.
    frames = [] if args.video else None

    # -------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------
    credited = []
    twist = []
    swings = []

    rates = []
    drops = []
    returns = []
    steps = []

    # -------------------------------------------------------------
    # EPISODES
    # -------------------------------------------------------------
    for i in range(args.episodes):

        episode_return, info, net_twist, worst_swing = rollout(
            env,
            policy,
            axis,
            seed=args.seed + i,
            frames=(
                frames
                if args.video and i == 0
                else None
            ),
            show=args.show,
        )

        credited.append(
            float(
                info.get(
                    "ep_revolutions",
                    0.0,
                )
            )
        )

        twist.append(
            net_twist / (2.0 * np.pi)
        )

        swings.append(
            worst_swing
        )

        rates.append(
            float(
                info.get(
                    "ep_rot_per_sec",
                    0.0,
                )
            )
        )

        drops.append(
            float(
                info.get(
                    "ep_dropped",
                    0.0,
                )
            )
        )

        steps.append(
            int(
                info.get(
                    "ep_steps",
                    0,
                )
            )
        )

        returns.append(
            episode_return
        )

        # Tell user which episode is currently visible.
        if args.show:
            print(
                f"\rEpisode "
                f"{i + 1}/{args.episodes}",
                end="",
                flush=True,
            )

    if args.show:
        print()

    # -------------------------------------------------------------
    # CLOSE ENV
    # -------------------------------------------------------------
    cube_cm = env.cube_half * 200
    max_steps = env.cfg.max_steps

    env.close()

    # -------------------------------------------------------------
    # NUMPY ARRAYS
    # -------------------------------------------------------------
    credited = np.asarray(
        credited,
        dtype=np.float64,
    )

    twist = np.asarray(
        twist,
        dtype=np.float64,
    )

    swings = np.asarray(
        swings,
        dtype=np.float64,
    )

    rates = np.asarray(
        rates,
        dtype=np.float64,
    )

    drops = np.asarray(
        drops,
        dtype=np.float64,
    )

    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    steps = np.asarray(
        steps,
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------
    n = len(credited)

    if n > 1:
        sem = (
            credited.std(ddof=1)
            / np.sqrt(n)
        )
    else:
        sem = 0.0

    trackable = (
        swings
        <= MAX_TRACKABLE_SWING
    )

    # -------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------
    print()

    print(
        f"policy: {label}   "
        f"episodes={n}   "
        f"randomize={args.randomize}   "
        f"cube {cube_cm:.1f} cm"
    )

    print(
        f"  revolutions (credited)   "
        f"{credited.mean():+.3f} "
        f"+- {sem:.3f} SEM"
        f"   [{credited.min():+.3f} .. "
        f"{credited.max():+.3f}]"
    )

    # -------------------------------------------------------------
    # SWING-TWIST
    # -------------------------------------------------------------
    if trackable.any():

        credited_trackable = (
            credited[trackable]
        )

        twist_trackable = (
            twist[trackable]
        )

        if (
            abs(
                credited_trackable.mean()
            ) > 1e-9
        ):
            ratio = (
                twist_trackable.mean()
                / credited_trackable.mean()
            )

            ratio_text = (
                f"   ratio {ratio:.3f}"
            )

        else:
            ratio_text = ""

        print(
            f"  revolutions (swing-twist)"
            f"{twist_trackable.mean():+.3f}"
            f"{ratio_text}"
            f"   over "
            f"{int(trackable.sum())}/{n} "
            f"trackable episodes"
        )

    else:

        print(
            "  revolutions (swing-twist)  "
            "n/a -- every episode passed "
            f"within "
            f"{np.degrees(MAX_TRACKABLE_SWING):.0f} "
            "deg of a flip"
        )

    if not trackable.all():

        print(
            f"  (excluded "
            f"{int((~trackable).sum())} "
            "episodes where the cube tipped "
            f"past "
            f"{np.degrees(MAX_TRACKABLE_SWING):.0f} "
            "deg and the twist stops being "
            "defined)"
        )

    # -------------------------------------------------------------
    # OTHER METRICS
    # -------------------------------------------------------------
    print(
        f"  degrees                  "
        f"{np.degrees(credited.mean() * 2 * np.pi):+.1f}"
        f"   (credited.mean() * 2 * np.pi)"
        f"):+.1f"
    )

    print(
        f"  rad/s                    "
        f"{rates.mean():+.4f}"
    )

    print(
        f"  return                   "
        f"{returns.mean():+.2f}"
    )

    print(
        f"  drop rate                "
        f"{drops.mean():.3f}"
        f"   mean steps "
        f"{steps.mean():.0f} / "
        f"{max_steps}"
    )

    # -------------------------------------------------------------
    # THRESHOLDS
    # -------------------------------------------------------------
    for threshold in (
        0.25,
        0.50,
        1.00,
        2.00,
    ):

        fraction = float(
            np.mean(
                credited >= threshold
            )
        )

        print(
            f"  >= {threshold:>4.2f} rev"
            f"             {fraction:.3f}"
        )

    success_fraction = float(
        np.mean(
            credited
            >= SPIN_CFG.success_revolutions
        )
    )

    print(
        f"  SUCCESS "
        f"(>= "
        f"{SPIN_CFG.success_revolutions:.1f}"
        f" rev)"
        f"      {success_fraction:.3f}"
    )

    # -------------------------------------------------------------
    # SAVE VIDEO
    # -------------------------------------------------------------
    if args.video and frames:

        import imageio

        imageio.mimwrite(
            args.video,
            frames,
            fps=50,
            quality=8,
        )

        print(
            f"  video -> {args.video}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------
    # POLICY SOURCE
    # -------------------------------------------------------------
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--run",
        type=int,
        default=None,
        choices=tuple(RUNS),
        help=(
            "a released checkpoint, "
            "with its own scene and closure"
        ),
    )

    group.add_argument(
        "--actor",
        default=None,
        help="an exported actor .npz",
    )

    group.add_argument(
        "--model",
        default=None,
        help="an SB3 SAC .zip model",
    )

    group.add_argument(
        "--gait",
        action="store_true",
        help="open-loop CEM baseline",
    )

    # -------------------------------------------------------------
    # ENVIRONMENT
    # -------------------------------------------------------------
    parser.add_argument(
        "--gait-file",
        default=None,
    )

    parser.add_argument(
        "--scene",
        default=None,
    )

    parser.add_argument(
        "--close-frac",
        type=float,
        default=None,
    )

    # -------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------
    parser.add_argument(
        "--episodes",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--randomize",
        action="store_true",
        help=(
            "domain randomisation + "
            "sensor noise"
        ),
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
    )

    # -------------------------------------------------------------
    # VIDEO
    # -------------------------------------------------------------
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "save the first episode "
            "as a video file"
        ),
    )

    # -------------------------------------------------------------
    # LIVE MUJOCO VIEWER
    # -------------------------------------------------------------
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "open the live MuJoCo viewer "
            "and display the policy moving "
            "the hand and cube"
        ),
    )

    args = parser.parse_args()

    main(args)