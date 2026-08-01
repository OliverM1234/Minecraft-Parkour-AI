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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------------------------
# Course layout constants - shared between the mission XML builder below and
# ParkourRewardWrapper, so they must stay in sync.
# ---------------------------------------------------------------------------
NUM_PLATFORMS = 12
PLATFORM_Y = 4  # block layer the platforms are drawn on
PLATFORM_TOP_Y = float(PLATFORM_Y + 1)  # agent's feet rest here when standing on one
PLATFORM_HALF_WIDTH = 1  # platforms are (2 * this + 1) blocks wide, for drift tolerance
PLATFORM_SPACING = 2  # z-distance between platform centers -> exactly a 1-block gap
BLOCK_TYPE = "quartz_block"
FALL_TOLERANCE = 0.5  # "fallen" once y drops this far below a platform's surface

# yaw=0 faces +Z in Minecraft's coordinate convention, matching the course
# direction below; pitch=30 looks down at the blocks instead of the horizon.
SPAWN_X, SPAWN_Y, SPAWN_Z = 0.5, PLATFORM_TOP_Y, 0.5
SPAWN_YAW, SPAWN_PITCH = 0.0, 30.0


def build_parkour_course():
    """A straight line of single-block-gap floating platforms along +Z.

    Returns (drawing_str, start_position) for MineDojoSim's drawing_str /
    start_position kwargs.
    """
    cuboids = []
    for i in range(NUM_PLATFORMS):
        z = i * PLATFORM_SPACING
        cuboids.append(
            f'<DrawCuboid x1="{-PLATFORM_HALF_WIDTH}" y1="{PLATFORM_Y}" z1="{z}" '
            f'x2="{PLATFORM_HALF_WIDTH}" y2="{PLATFORM_Y}" z2="{z}" type="{BLOCK_TYPE}"/>'
        )
    drawing_str = "".join(cuboids)
    start_position = {
        "x": SPAWN_X, "y": SPAWN_Y, "z": SPAWN_Z, "yaw": SPAWN_YAW, "pitch": SPAWN_PITCH,
    }
    return drawing_str, start_position


def make_base_env():
    """Raw MineDojo sim: a void world (single bedrock layer) with the course
    drawn into it. Built from MineDojoSim directly rather than minedojo.make(),
    since make()'s NNActionSpaceWrapper packs jump/sneak/sprint into a single
    mutually-exclusive slot and can't express "jump AND sprint" in one step -
    exactly the combo a parkour bot needs for sprint-jumping a gap.
    """
    drawing_str, start_position = build_parkour_course()
    return MineDojoSim(
        image_size=(160, 256),
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
    """Discrete(3): 0 forward, 1 forward+jump, 2 forward+jump+sprint.

    Camera lock falls out of this for free: "camera" is never touched, so the
    fixed pitch/yaw set at spawn (start_position) never changes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, act):
        noop = self.env.action_space.no_op()
        noop["forward"] = 1
        if act == 1:
            noop["jump"] = 1
        elif act == 2:
            noop["jump"] = 1
            noop["sprint"] = 1
        return noop

    def reverse_action(self, action):
        raise NotImplementedError


class ParkourRewardWrapper(gym.Wrapper):
    """Dense forward-progress reward, landing bonus, and fall penalty/termination.

    Wraps the action-wrapped env (not yet grayscaled) so it still has access to
    the full obs dict's location_stats for its own bookkeeping.
    """

    STEP_PENALTY = -0.01
    LANDING_BONUS = 10.0
    FALL_PENALTY = -100.0

    def __init__(self, env):
        super().__init__(env)
        self._prev_z = None
        self._best_platform_idx = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # AgentStartPlacement in the mission XML isn't reliably re-applied on
        # repeat missions within the same running Minecraft instance (observed:
        # first-ever reset spawns correctly, later resets can land far from the
        # course). Force it explicitly every time rather than trust the XML alone.
        obs, _, _, _ = self.env.execute_cmd(
            f"/tp @p {SPAWN_X} {SPAWN_Y} {SPAWN_Z} {SPAWN_YAW} {SPAWN_PITCH}"
        )
        self._prev_z = float(obs["location_stats"]["pos"][2])
        self._best_platform_idx = 0
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        x, y, z = (float(v) for v in obs["location_stats"]["pos"])

        reward = self.STEP_PENALTY
        reward += z - self._prev_z  # dense continuous form of "+1 per block forward"
        self._prev_z = z

        fell = y < PLATFORM_TOP_Y - FALL_TOLERANCE
        if fell:
            reward += self.FALL_PENALTY
            done = True
        else:
            platform_idx = round(z / PLATFORM_SPACING)
            on_a_platform = abs(z - platform_idx * PLATFORM_SPACING) < 0.5
            if on_a_platform and platform_idx > self._best_platform_idx:
                reward += self.LANDING_BONUS
                self._best_platform_idx = platform_idx
                if self._best_platform_idx >= NUM_PLATFORMS - 1:
                    done = True  # reached the last platform

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
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make_env():
    env = make_base_env()
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
    args = parser.parse_args()

    CHECKPOINT_DIR = "checkpoints"
    LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "parkour_ppo_latest.zip")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # A single Minecraft instance - this laptop can't run several in parallel.
    env = DummyVecEnv([make_env])

    if os.path.exists(LATEST_CHECKPOINT):
        print(f"Resuming from {LATEST_CHECKPOINT}")
        model = PPO.load(LATEST_CHECKPOINT, env=env, device="cpu")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cpu",  # integrated GPU only, no CUDA
            verbose=1,
            tensorboard_log="tb_logs",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=2000, save_path=CHECKPOINT_DIR, name_prefix="parkour_ppo"
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
    finally:
        model.save(LATEST_CHECKPOINT)
        env.close()
