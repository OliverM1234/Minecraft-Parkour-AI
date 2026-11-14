"""
Minecraft parkour RL bot: MineDojo (legacy gym==0.21.0) + Stable-Baselines3 PPO.

Run:
    MINEDOJO_HEADLESS=1 /home/oliver-m/minedojo_env/bin/python -u train_parkour.py [--total-timesteps N]

Requires stable-baselines3==1.8.0 (the last pre-gymnasium release) + torch (CPU build).
"""
import os

os.environ.setdefault("MINEDOJO_HEADLESS", "1")

import argparse
import numpy as np
import cv2
import gym
from gym import spaces
from gym.wrappers import TimeLimit

from minedojo.sim import MineDojoSim

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# PyTorch's default intra-op parallelism spawns threads across every logical
# core for each forward/backward pass. On this small a CNN policy the thread
# synchronization overhead isn't worth it, and it directly competes with the
# Minecraft JVM(s) for CPU - the same contention the --n-envs
# measurement found. Applies to watch_parkour.py's inference too, since it
# imports this module.
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Course layout constants - shared between the mission XML builder below and
# ParkourRewardWrapper, so they must stay in sync.
# ---------------------------------------------------------------------------
# (x, z, half_width_x). half_width_x only ever widens a platform along X -
# z is always exactly 1 block wide, so the deliberate z-gap between any two
# platforms never shrinks no matter how wide a platform gets in x.
PLATFORMS = [
    (0.0, 0.0, 0),    # 0: spawn
    (0.0, 2.0, 0),    # 1: forward 2 (straight, cold start - confirmed working)
    (-1.0, 5.0, 0),   # 2: left 1, forward 3 (diagonal - confirmed working)
    (-1.0, 9.0, 1),   # 3: forward 4 (straight, but sprinting by now - see note)
    (0.0, 13.0, 0),   # 4: right 1, forward 4 (diagonal, sprinting - projected)
    (1.0, 17.0, 0),   # 5: right 1, forward 4 (diagonal, sprinting - projected)
    (1.0, 21.0, 1),   # 6: forward 4 (straight, sprinting - projected)
    (0.0, 25.0, 0),   # 7: left 1, forward 4 (diagonal, sprinting - projected)
]
NUM_PLATFORMS = len(PLATFORMS)

# Hop distances are NOT uniform, and this isn't guesswork - measured against a
# real instance. Momentum/sprint state carries across hops: hop0 (a true cold
# start, zero prior velocity) covers only dz=2 before landing. Every hop after
# that inherits sprint momentum and covers much more ground per hop - hop2 (a
# plain forward hop, no strafe input at all) sailed straight through a dz=2
# target still 1+ block above platform height and didn't return to platform
# height until dz~4. Hops 3 onward use dz=4 to match. The two diagonal hops
# already confirmed (hop1) used dz=3 - that was still in the "not fully hot
# yet" transitional zone right after hop0; hops past that are projected to
# behave like the confirmed hot *straight* hop (dz=4) since no diagonal hop
# from a fully-hot state has been measured directly. If training or a replay
# shows platforms 4-7 still don't land, re-measure them the same way platform
# 3 was (run the preceding hops, then let the next hop fully play out past a
# fall to see where it naturally returns to platform height).
#
# Holding forward+strafe+jump+sprint (diagonal) is forward-biased, not a 45
# degree path - dz covers roughly 2-3x whatever dx does for the same hold
# duration, not a 1:1 split, and there's no discrete action for an in-between
# angle. Platforms 3 and 6 are widened in x (half_width_x=1) because a
# straight hop's landing point drifts sideways from the diagonal hop's
# residual momentum right before it, even with zero strafe input on the
# straight hop itself.
PLATFORM_Y = 4  # block layer the platforms are drawn on
PLATFORM_TOP_Y = float(PLATFORM_Y + 1)  # agent's feet rest here when standing on one
BLOCK_TYPE = "quartz_block"
FALL_TOLERANCE = 0.5  # "fallen" once y drops this far below a platform's surface

# yaw=0 faces +Z in Minecraft's coordinate convention, matching the course
# direction below; pitch=30 looks down at the blocks instead of the horizon.
# Minecraft block centers are offset by +0.5 from their root coordinates.
SPAWN_X, SPAWN_Y, SPAWN_Z = 0.5, PLATFORM_TOP_Y, 0.5
SPAWN_YAW, SPAWN_PITCH = 0.0, 30.0


def build_parkour_course():
    """Builds floating platforms matching the custom (x, z) coordinate list.

    Returns (drawing_str, start_position) for MineDojoSim's drawing_str /
    start_position kwargs.
    """
    cuboids = []
    for x, z, half_width_x in PLATFORMS:
        # Cast all XML coordinate values to int to satisfy XML schema validation.
        # z1==z2 always (exactly 1 block deep) - only x ever widens.
        x1_int = int(x - half_width_x)
        x2_int = int(x + half_width_x)
        z1_int = z2_int = int(z)
        y_int = int(PLATFORM_Y)

        cuboids.append(
            f'<DrawCuboid x1="{x1_int}" y1="{y_int}" z1="{z1_int}" '
            f'x2="{x2_int}" y2="{y_int}" z2="{z2_int}" type="{BLOCK_TYPE}"/>'
        )
    drawing_str = "".join(cuboids)
    start_position = {
        "x": SPAWN_X, "y": SPAWN_Y, "z": SPAWN_Z, "yaw": SPAWN_YAW, "pitch": SPAWN_PITCH,
    }
    return drawing_str, start_position


def make_base_env(image_size=(160, 256)):
    """Raw MineDojo sim: a void world (single bedrock layer) with the course
    drawn into it. Built from MineDojoSim directly rather than minedojo.make(),
    since make()'s NNActionSpaceWrapper packs jump/sneak/sprint into a single
    mutually-exclusive slot and can't express "jump AND sprint" in one step -
    exactly the combo a parkour bot needs for sprint-jumping a gap.

    Default image_size (160,256) matches what watch_parkour.py's
    VideoRecorderWrapper expects for its raw-frame MP4 capture - don't lower
    this default, pass a smaller image_size explicitly at the call site
    instead (see make_env() below), or recorded videos get quietly downscaled.
    """
    drawing_str, start_position = build_parkour_course()
    return MineDojoSim(
        image_size=image_size,
        generate_world_type="flat",
        flat_world_seed_string="1;7;1",  # one bedrock layer only -> void above it
        drawing_str=drawing_str,
        start_position=start_position,
        start_time=6000,  # noon
        allow_time_passage=False,  # keep lighting constant across the episode
        allow_mob_spawn=False,
        seed=0,
    )


class ParkourActionWrapper(gym.ActionWrapper):
    """Discrete(5):
    0: Forward
    1: Forward + Jump
    2: Forward + Jump + Sprint
    3: Forward + Left + Jump + Sprint
    4: Forward + Right + Jump + Sprint

    Camera lock falls out of this for free: "camera" is never touched, so the
    fixed pitch/yaw set at spawn (start_position) never changes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)

    def action(self, act):
        noop = self.env.action_space.no_op()
        noop["forward"] = 1

        if act == 1:
            noop["jump"] = 1
        elif act == 2:
            noop["jump"] = 1
            noop["sprint"] = 1
        elif act == 3:
            noop["left"] = 1
            noop["jump"] = 1
            noop["sprint"] = 1
        elif act == 4:
            noop["right"] = 1
            noop["jump"] = 1
            noop["sprint"] = 1

        return noop

    def reverse_action(self, action):
        raise NotImplementedError


class ParkourRewardWrapper(gym.Wrapper):
    """Potential-based progress reward (shrinking distance to the next target
    platform), a landing bonus, and fall penalty/termination.

    Wraps the action-wrapped env (not yet grayscaled) so it still has access to
    the full obs dict's location_stats for its own bookkeeping.
    """

    STEP_PENALTY = -0.01
    LANDING_BONUS = 10.0
    FALL_PENALTY = -100.0
    # Kept below each platform's true half-extent on both axes, else a
    # trajectory clipping a platform's corner without being supported by it
    # can satisfy the landing check before the fall check catches up.
    EDGE_MARGIN = 0.1
    # y must be this close to PLATFORM_TOP_Y to count as "landed" rather than
    # still mid-arc above/near the target platform's (x, z) column.
    GROUNDED_TOLERANCE = 0.1

    def __init__(self, env):
        super().__init__(env)
        self._best_platform_idx = 0
        self._prev_dist_to_target = None

    def _dist_to_platform(self, x, z, idx):
        tx, tz, _ = PLATFORMS[idx]
        # Minecraft block centers sit at +0.5 relative to their root coordinate.
        # Used only for the continuous reward-shaping term below, so a circular
        # approximation of a possibly-non-square footprint is fine here.
        return float(np.sqrt((x - (tx + 0.5)) ** 2 + (z - (tz + 0.5)) ** 2))

    def _landed_on(self, x, z, idx):
        # Axis-aligned box check matching the platform's actual (possibly
        # x-widened, always z=1-block) footprint - a circular radius would be
        # wrong here since platforms aren't all square.
        tx, tz, half_width_x = PLATFORMS[idx]
        cx, cz = tx + 0.5, tz + 0.5
        return (
            abs(x - cx) < (half_width_x + 0.5 - self.EDGE_MARGIN)
            and abs(z - cz) < (0.5 - self.EDGE_MARGIN)
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # AgentStartPlacement in the mission XML isn't reliably re-applied on
        # repeat missions within the same running Minecraft instance (observed:
        # first-ever reset spawns correctly, later resets can land far from the
        # course). Force it explicitly every time rather than trust the XML alone.
        obs, _, _, _ = self.env.execute_cmd(
            f"/tp @p {SPAWN_X} {SPAWN_Y} {SPAWN_Z} {SPAWN_YAW} {SPAWN_PITCH}"
        )
        self._best_platform_idx = 0
        x, _, z = (float(v) for v in obs["location_stats"]["pos"])
        self._prev_dist_to_target = self._dist_to_platform(x, z, 1)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        x, y, z = (float(v) for v in obs["location_stats"]["pos"])

        reward = self.STEP_PENALTY

        fell = y < PLATFORM_TOP_Y - FALL_TOLERANCE
        if fell:
            reward += self.FALL_PENALTY
            done = True
        else:
            next_idx = self._best_platform_idx + 1
            if next_idx < NUM_PLATFORMS:
                dist = self._dist_to_platform(x, z, next_idx)
                # Potential-based shaping: reward = reduction in distance to the
                # next target since last step. Generalizes "+1 per block forward"
                # to a course that moves in x as well as z - a plain dz reward
                # gave zero signal for the lateral movement 3 of the 4 hops need.
                reward += self._prev_dist_to_target - dist
                self._prev_dist_to_target = dist

                grounded = abs(y - PLATFORM_TOP_Y) < self.GROUNDED_TOLERANCE
                if grounded and self._landed_on(x, z, next_idx):
                    reward += self.LANDING_BONUS
                    self._best_platform_idx = next_idx
                    if self._best_platform_idx >= NUM_PLATFORMS - 1:
                        done = True  # course completed
                    else:
                        self._prev_dist_to_target = self._dist_to_platform(x, z, next_idx + 1)

        info = dict(info, x=x, y=y, z=z, fell=fell, platform_idx=self._best_platform_idx)
        return obs, reward, done, info


class ActionRepeatWrapper(gym.Wrapper):
    """Holds the chosen action for `repeat` real ticks (default 4).

    Sits outside ParkourRewardWrapper so fall-detection/progress reward see
    every tick, not just 1-in-4; sits inside GrayscaleResizeObservation so the
    grayscale/resize conversion runs once per agent decision instead of once
    per repeated tick.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        obs, done, info = None, False, {}
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayscaleResizeObservation(gym.ObservationWrapper):
    """obs["rgb"] (3,H,W uint8) -> a single grayscale (size,size,1) uint8 frame.

    Emits channel-LAST, like a normal Gym image env, so SB3's VecTransposeImage
    converts it to channel-first automatically; emitting channel-first
    ourselves here would risk SB3 double-transposing a square image silently.
    """

    def __init__(self, env, size=64):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )

    def observation(self, obs):
        frame = obs["rgb"].transpose(1, 2, 0)  # (3,H,W) -> (H,W,3), RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return frame[:, :, None]


def make_env():
    # Render natively at the policy's actual input size (64x64) instead of
    # 160x256 then downscaling - cuts Java's rendering/IPC-transfer load.
    # watch_parkour.py calls make_base_env() with no argument, so its video
    # capture keeps the full 160x256 default untouched.
    env = make_base_env(image_size=(64, 64))
    env = ParkourActionWrapper(env)
    env = ParkourRewardWrapper(env)
    env = ActionRepeatWrapper(env, repeat=4)
    env = GrayscaleResizeObservation(env, size=64)
    env = TimeLimit(env, max_episode_steps=200)
    # Monitor must be outermost so it also sees TimeLimit's own truncation,
    # not just the reward wrapper's fall-termination.
    env = Monitor(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Absolute cumulative-timestep target (not additional steps) - "
        "safe to rerun with a bigger number later, training resumes from the checkpoint.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel Minecraft instances. 1 uses DummyVecEnv (in-process); "
        ">1 uses SubprocVecEnv. Measured: 2 instances contend for CPU on this "
        "hardware rather than running truly in parallel - no throughput win, "
        "default is 1.",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR = "checkpoints"
    LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "parkour_ppo_latest.zip")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    N_ENVS = args.n_envs
    if N_ENVS > 1:
        # SubprocVecEnv spawns each Minecraft instance in its own process (bypasses
        # the GIL); MineDojo's InstanceManager allocates each its own port, so these
        # don't collide with each other.
        env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env])

    if os.path.exists(LATEST_CHECKPOINT):
        print(f"Resuming from {LATEST_CHECKPOINT}")
        # n_epochs overridden explicitly: a checkpoint saved under the old
        # n_epochs=10 would otherwise silently keep training at that value.
        model = PPO.load(LATEST_CHECKPOINT, env=env, device="cpu", n_epochs=4)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            device="cpu",  # integrated GPU only, no CUDA
            verbose=1,
            tensorboard_log="tb_logs",
        )

    # save_freq counts calls to _on_step(), which fire once per N_ENVS real
    # timesteps under a VecEnv - divide so checkpoints still land every ~2000
    # timesteps rather than silently doubling to ~4000 now that N_ENVS=2.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(2000 // N_ENVS, 1), save_path=CHECKPOINT_DIR, name_prefix="parkour_ppo"
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=True,  # tqdm/rich; negligible overhead, updates once per step
        )
    finally:
        model.save(LATEST_CHECKPOINT)
        env.close()