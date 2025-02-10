import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def run(self, is_training = True, render = False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)

        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        reward_per_episode = []

        policy_dqn = DQN(num_state, num_action).to_device(device)

        if is_training:
            memory = ReplayMemory(10000)

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                # Accumulate reward
                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                # Move to new state
                state = new_state
                
            reward_per_episode.append(episode_reward)