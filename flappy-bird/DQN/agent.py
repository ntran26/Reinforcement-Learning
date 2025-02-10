import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']

    def run(self, is_training = True, render = False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)

        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_state, num_action).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # add another dimension to be compatible with pytorch
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # Accumulate reward
                episode_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                # Move to new state
                state = new_state
                
            reward_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == "__main__":
    agent = Agent("CartPole")
    agent.run(is_training=True, render=True)