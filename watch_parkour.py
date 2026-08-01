import os
import cv2
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure headless mode is set before MineDojo client initializes
os.environ.setdefault("MINEDOJO_HEADLESS", "1")

# Import environment definitions from train_parkour.py
from train_parkour import (
    make_base_env,
    ParkourActionWrapper,
    ParkourRewardWrapper,
    ActionRepeatWrapper,
    GrayscaleResizeObservation,
)


class VideoRecorderWrapper(gym.Wrapper):
    """
    Captures raw, full-resolution RGB frames (160x256) directly from MineDojo
    at 20 FPS before action repeating or grayscaling occur.
    """
    def __init__(self, env, filename="parkour_run.mp4", fps=20):
        super().__init__(env)
        self.filename = filename
        self.fps = fps
        self.frames = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._capture_frame(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Do NOT reset self.frames here, as DummyVecEnv calls reset() 
        # automatically at episode end before we get to save it!
        if not self.frames:
            self._capture_frame(obs)
        return obs

    def _capture_frame(self, obs):
        frame_rgb = obs["rgb"].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.frames.append(frame_bgr)

    def close(self):
        if self.frames:
            print(f"Saving {len(self.frames)} frames to {self.filename}...")
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print("Video saved successfully!")
        super().close()


def make_eval_env(video_name="parkour_run.mp4"):
    """
    Rebuilds the training env pipeline, placing VideoRecorderWrapper inside
    ActionRepeatWrapper so every 1/20s tick is recorded, but inside 
    GrayscaleResizeObservation so the policy receives 64x64 grayscale frames.
    """
    env = make_base_env()
    env = ParkourActionWrapper(env)
    env = ParkourRewardWrapper(env)

    # Record 20 FPS raw RGB frames here
    env = VideoRecorderWrapper(env, filename=video_name, fps=20)

    env = ActionRepeatWrapper(env, repeat=4)
    env = GrayscaleResizeObservation(env, size=64)
    env = TimeLimit(env, max_episode_steps=200)
    return env


if __name__ == "__main__":
    CHECKPOINT_PATH = "checkpoints/parkour_ppo_latest.zip"

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at '{CHECKPOINT_PATH}'")

    print("Initializing environment...")
    # Wrap in DummyVecEnv to match SB3 1.8.0 VecTransposeImage parity
    eval_env = DummyVecEnv([lambda: make_eval_env("parkour_run.mp4")])

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = PPO.load(CHECKPOINT_PATH, env=eval_env, device="cpu")

    obs = eval_env.reset()
    done = False
    episode_reward = 0.0

    print("Running evaluation episode...")
    while not done:
        # deterministic=True disables PPO action sampling for greedy evaluation
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done_vec, infos = eval_env.step(action)
        
        done = done_vec[0]
        episode_reward += reward[0]
        info = infos[0]

    # Report results
    if info.get("fell", False):
        print(f"Agent fell! Highest platform reached: {info.get('platform_idx', 0)}")
    else:
        print(f"Episode finished! Highest platform reached: {info.get('platform_idx', 0)}")
    print(f"Total Episode Reward: {episode_reward:.2f}")

    # Closing eval_env triggers VideoRecorderWrapper.close() to write the MP4
    eval_env.close()