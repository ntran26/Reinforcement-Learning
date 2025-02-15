import os
import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# For printing date time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving runs info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# For generating plots as images and save them to file
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameters_set = hyperparameter_set

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']

        # Neural network
        self.loss_fn = nn.MSELoss()     # Mean squared error
        self.optimizer = None           # Placeholer to initialize optimizer

        # Path to runs info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')

    def run(self, is_training = True, render = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        policy_dqn = DQN(num_state, num_action, self.fc1_nodes).to(device)

        # Initialize list to store rewards
        reward_per_episode = []

        last_graph_update_time = datetime.now()

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            # Sync policy to target network
            target_dqn = DQN(num_state, num_action, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Track number of steps taken
            step_count = 0

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            # Initialize epsilon history
            epsilon_history = []

            # Track best rewards
            best_reward = -99999999
        
        else:       # if not training, load trained model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Begin an episode
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0

            while (not terminated and episode_reward < self.stop_on_reward):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # add another dimension to be compatible with pytorch
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # Accumulate reward
                episode_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1     # increase step count

                # Move to new state
                state = new_state
            
            # Keep track of reward collected per episode
            reward_per_episode.append(episode_reward)

            # Save the model when reached new best reward
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"New best reward: {episode_reward:0.1f} ({(episode_reward - best_reward)}%)"
                    print(log_message)

                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

            # If enough experience has been collected
            if len(memory) > self.mini_batch_size:

                # sample from memory
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Update epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def optimize(self, mini_batch, policy, target):
        # for state, action, new_state, reward, terminated in mini_batch:
        #     if terminated:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor * target(new_state).max()
            
        #     current_q = policy(state)
        
        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors 
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)
    
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # calculate target Q value (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor * target(new_states).max(dim=1)[0]
            '''
                target(new_states) => tensor([[1,2,3],[4,5,6]])
                .max(dim=1)        => torch.return_types.max(values = tensor([3,6]), indices = tensor([3,0,0,1]))
                [0]                => tensor([3,6])
            '''
        
        # calculate Q values from current policy
        current_q = policy(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy(new_states)                       => tensor([[1,2,3],[4,5,6]])
            actions.unsqueeze(dim=1) 
            .gather(1, actions.unsqueeze(dim=1))     => 
            .squeeze()                               =>
        '''

        # Compute loss for the mini batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()      # clear gradients
        loss.backward()                 # compute gradient (backpropagation)
        self.optimizer.step()           # update network parameters (weights and biases)
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards vs episodes
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0,x-99):(x+1)])
        plt.subplot(121)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)

        # Plot epsilon decay vs episodes
        plt.subplot(122)
        plt.xlabel("Timesteps")
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    agent = Agent("cartpole1")

    train = False
    if train:   # Train the model
        agent.run(is_training=True)
    else:       # Test with existing model
        agent.run(is_training=False, render=True)