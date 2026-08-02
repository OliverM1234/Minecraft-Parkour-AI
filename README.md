# Minecraft-Parkour-AI

A Reinforcement Learning (RL) project that trains an autonomous agent to navigate a custom-built parkour course in Minecraft. 

This project utilizes **MineDojo** as the simulation environment and **Stable-Baselines3** (PPO) to teach the agent complex movement combinations, such as sprint-jumping across floating platforms.

## 🧠 Overview

Training an AI to do parkour in Minecraft requires precise timing and specific action combinations. Standard environment wrappers often make actions mutually exclusive (e.g., you can jump OR sprint). This project implements custom action and reward wrappers to allow the agent to execute simultaneous actions (like forward + sprint + jump) and provides dense reward shaping to encourage forward progress and successful landings while heavily penalizing falls.

### Key Features
* **Custom Mission Generation:** Dynamically builds a void-world parkour course using MineDojo's XML string drawing capabilities.
* **Specialized Action Space:** A custom `Discrete(5)` action wrapper that specifically allows for complex parkour maneuvers.
* **Reward Shaping:** Implements continuous forward-progress rewards, landing bonuses for hitting the next platform, and immediate termination/penalties for falling.
* **Observation Processing:** Grayscales and resizes the agent's POV to a 64x64 frame and implements Action Repeat (frame skipping) to optimize training speed for CNN policies.

---

## 🛠 Requirements & Installation

**Important Note on Dependencies:** MineDojo requires legacy versions of `gym` and `stable-baselines3`. Using modern versions (like `gymnasium`) will cause compatibility errors.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/OliverM1234/Minecraft-Parkour-AI.git](https://github.com/OliverM1234/Minecraft-Parkour-AI.git)
   cd Minecraft-Parkour-AI
