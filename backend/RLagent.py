#!/usr/bin/env python
# coding: utf-8

# In[5]:


import gymnasium as gym
from gymnasium import spaces
import os
import time
import numpy as np
import torch
import torch.nn as nn
from IPython.display import clear_output
import math
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import matplotlib.pyplot as plt
import random
import gym


# In[2]:


class Connect4Env(gym.Env):
    def __init__(self):
        super(Connect4Env, self).__init__()
        self.action_space = spaces.Discrete(7)  # 7 possible actions: one for each column
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int32)  # Corrected dtype to np.int32
        self.board = self.create_board()
        self.current_player = 1

    def create_board(self):
        return [[0 for _ in range(7)] for _ in range(6)]

    def make_move(self, column):
        for row in reversed(range(6)):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                return True
        return False  # Column is full

    def check_win(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 0:
                    continue
                for dr, dc in directions:
                    if all(0 <= row + i*dr < 6 and 0 <= col + i*dc < 7 and self.board[row + i*dr][col + i*dc] == self.board[row][col] for i in range(4)):
                        return True
        return False

    def check_full(self):
        return all(self.board[0][col] != 0 for col in range(7))

    def display_board(self):
        print("\n  0 1 2 3 4 5 6")
        print("  ------------")
        for row in self.board:
            print('|' + ' '.join('O' if cell == 1 else 'X' if cell == 2 else ' ' for cell in row) + '|')
        print("  ------------")

    def step(self, action):
        done = False
        truncated = False  # Indicates if the episode was truncated (not used here but required by the Gym API)
        reward = 0
        info = {}  # Initialize an empty info dictionary

        # Ensure action is an integer, handling both scalar and array-wrapped actions
        action = action.item() if isinstance(action, np.ndarray) else int(action)
    
        # Attempt to make a move on the board
        valid_move = self.make_move(action)
        if not valid_move:
            reward = -10  # Penalize invalid moves
            done = True  # End the episode on an invalid move
            # You can add relevant info regarding invalid moves if needed
            info['invalid_move'] = True
            return np.array(self.board), reward, done, truncated, info

        # Check for a win condition
        if self.check_win():
            done = True
            reward = 1  # Reward for winning
            info['win'] = True  # Optionally, add win info

        # Check if the board is full, indicating a tie
        elif self.check_full():
            done = True
            reward = 0.5  # Reward for a tie (you can adjust this as needed)
            info['tie'] = True  # Optionally, add tie info

        # Switch the current player
        self.current_player = 3 - self.current_player

        # Return the updated board state, reward, done flag, truncated flag, and info dictionary
        return np.array(self.board), reward, done, truncated, info


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_state(self):
        """
        Returns the current state of the game board.

        :return: The current game board as a numpy array.
        """
        return np.array(self.board)

    def set_state(self, state):
        """
        Sets the game board to the given state.

        :param state: A numpy array representing the state to set the game board to.
        """
        if state.shape == (6, 7) and np.issubdtype(state.dtype, np.integer):
            self.board = state.tolist()
        else:
            raise ValueError("Invalid state shape or dtype. State should be a numpy array of shape (6, 7) with integer values.")

    def reset(self, seed=None, return_info=False, options=None):
        # It's a good practice to include seed handling if your environment uses randomness
        self.seed(seed)
        
        self.board = self.create_board()
        self.current_player = 1
        obs = np.array(self.board)

        # Prepare an empty info dictionary
        info = {}

        # The reset method should now return a tuple of (obs, info)
        return obs, info

    def render(self, mode='human', sleep_time=1):
        clear_output(wait=True)  # Clear the output to create an animation effect
        print("\n  0 1 2 3 4 5 6")
        print("  ------------")
        for row in self.board:
            print('|' + ' '.join('O' if cell == 1 else 'X' if cell == 2 else ' ' for cell in row) + '|')
        print("  ------------")
        time.sleep(sleep_time)  # Pause a bit to see the changes


# In[4]:


if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA (GPU) available, using CUDA.")
else:
    device = 'cpu'
    print("CUDA (GPU) not available, using CPU.")

env = Connect4Env()


# In[ ]:





# In[ ]:


class DistributionalDuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_atoms):
        super(DistributionalDuelingNetwork, self).__init__()
        self.num_atoms = num_atoms

        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        # Value and advantage stream modifications for distributional output
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim * num_atoms)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantages = self.advantage_stream(features).view(-1, self.action_size, self.num_atoms)
        q_vals = values + (advantages - advantages.mean(1, keepdim=True))
        return F.softmax(q_vals, dim=2)  # Apply softmax along the atom dimension


# In[ ]:


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


# In[ ]:


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, error):
        self.buffer.append((state, action, reward, next_state, done))
        # Prioritize more recent experiences
        priority = (error + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        probabilities = np.array(self.priorities) ** beta
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        # Importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            self.priorities[idx] = priority


# In[53]:


class SelfPlayCallback(BaseCallback):
    def __init__(self, save_path, update_interval, verbose=1):
        super(SelfPlayCallback, self).__init__(verbose)
        self.save_path = save_path
        self.update_interval = update_interval
        self.best_model = None

    def _on_step(self) -> bool:
        if self.n_calls % self.update_interval == 0:
            self.model.save(self.save_path.format(step=self.n_calls))
            self.best_model = self.model.load(self.save_path.format(step=self.n_calls))
            if self.verbose:
                print("Self-play updated model at step: ", self.n_calls)
        return True


# In[ ]:


def visualize_q_values(model, state):
    with torch.no_grad():
        q_values = model(state).cpu().numpy()
    actions = np.arange(len(q_values[0]))
    
    plt.figure(figsize=(10, 6))
    plt.bar(actions, q_values[0])
    plt.xlabel('Actions')
    plt.ylabel('Q-values')
    plt.title('Q-values for Different Actions')
    plt.xticks(actions)
    plt.show()


# In[54]:


def undo_move(current_state, history):
    if history:
        return history.pop()  # Remove the last state from history and return it
    return current_state  # If history is empty, return the current state unchanged

# Real-time Hints Functionality
def provide_hint(model, current_state):
    with torch.no_grad():
        q_values = model(current_state).cpu().numpy()
    recommended_action = np.argmax(q_values)
    return recommended_action  # Suggest the action with the highest Q-value


# In[57]:


writer = SummaryWriter(log_dir="./logs")

# During your training loop, you can log various metrics
def log_performance(writer, episode, total_reward, win_rate, loss):
    writer.add_scalar('Total Reward/Episode', total_reward, episode)
    writer.add_scalar('Win Rate/Episode', win_rate, episode)
    writer.add_scalar('Loss/Episode', loss, episode)

# Usage example (assuming 'episode', 'total_reward', 'win_rate', and 'loss' are defined in your training loop):
# log_performance(writer, episode, total_reward, win_rate, loss)

writer.close()  # Don't forget to close the writer when you're done logging


# In[ ]:


class OpponentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OpponentModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.layers(state)

def simulate_opponent_response(agent, opponent_model, state):
    with torch.no_grad():
        opponent_action_prob = opponent_model(state)
    # Simulate opponent's action based on the probability distribution
    opponent_action = torch.multinomial(opponent_action_prob, 1).item()
    # Simulate the environment's response to the opponent's action
    next_state, reward, done, _ = agent.env.step(opponent_action)
    return next_state, reward, done


# In[ ]:


# Example of incorporating opponent modeling into the decision-making process
def choose_action_with_opponent_modeling(agent, state, opponent_model):
    best_action = None
    best_value = -float('inf')
    for action in range(agent.action_space.n):
        # Simulate the agent's action
        simulated_next_state, _, _ = agent.env.step(action)
        # Simulate the opponent's response
        next_state_after_opponent, _, _ = simulate_opponent_response(agent, opponent_model, simulated_next_state)
        # Evaluate the state after the opponent's response
        value = agent.evaluate_state(next_state_after_opponent)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


# In[ ]:


# Main training loop
def train(agent, env, num_episodes):
    writer = SummaryWriter(log_dir="./logs")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)  # Modify to incorporate opponent modeling
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward
            state = next_state

        # Log performance, update target network, etc.
        log_performance(writer, episode, total_reward, agent.win_rate(), agent.loss())

    writer.close()


# In[ ]:


class YourAgentClass:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DistributionalDuelingNetwork(state_dim, action_dim, num_atoms=51)  # Assume num_atoms is 51 for Distributional RL
        self.target_model = DistributionalDuelingNetwork(state_dim, action_dim, num_atoms=51)
        self.buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.model(state).argmax(1).item()

    def learn(self, batch_size, gamma=0.99, beta=0.4):
        # Implementation of the learning process
        pass  # Complete this method according to your RL algorithm

    def evaluate_state(self, state):
        # Placeholder function for evaluating a state's value
        pass


# In[ ]:


def train(agent, env, num_episodes):
    writer = SummaryWriter()
    rewards_history = deque(maxlen=100)
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            error = np.abs(reward - agent.evaluate_state(state))
            agent.buffer.add(state, action, reward, next_state, done, error)
            state = next_state
            total_reward += reward

            if len(agent.buffer) > batch_size:
                agent.learn(batch_size)
        
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Average Reward: {avg_reward}")
        writer.add_scalar('Reward', total_reward, episode)

        if episode % 20 == 0:
            agent.update_target_model()
    
    writer.close()


# In[ ]:


# Assuming you have an environment `env` with `state_dim` and `action_dim`
agent = YourAgentClass(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
train(agent, env, num_episodes=1000)

