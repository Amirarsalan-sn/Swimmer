import os
import gc
import torch
import pygame
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

# for reproducible results:
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SquareActivation(nn.Module):
    def __init__(self):
        super(SquareActivation, self).__init__()

    def forward(self, x):
        return x.pow(2)


class PPOMemory:
    def __init__(self):
        self.states = deque()
        self.actions = deque()
        self.action_logs = deque()
        self.rewards = deque()
        self.next_states = deque()
        self.dones = deque()

    def store(self, state, action, action_log, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_logs.append(action_log)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def return_samples(self):
        indices = [i for i in range(len(self.dones))]

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(
            device)
        actions = torch.stack(
            [torch.as_tensor(self.actions[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        action_logs = torch.stack(
            [torch.as_tensor(self.action_logs[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, action_logs, rewards, next_states, dones

    def delete_mem(self):
        del self.states
        del self.actions
        del self.rewards
        del self.next_states
        del self.dones

        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.next_states = deque()
        self.dones = deque()


class PolicyNet(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(PolicyNet, self).__init__()

        self.FN = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True)
        )

        self.mean = nn.Sequential(
            nn.Linear(16, num_actions)
        )

        self.std = nn.Sequential(
            nn.Linear(16, num_actions),
            SquareActivation()
        )

        # He initialization.
        for layer in [self.FN]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

        for layer in [self.mean]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

        for layer in [self.std]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        mid_x = self.FN(x)
        mean = self.mean(mid_x)
        std = self.std(mid_x)

        return torch.cat([mean, std], dim=1).to(device)


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()

        self.FN = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        # He initialization.
        for layer in [self.FN]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FN(x)


class PPOClip:
    def __init__(self, epsilon, discount_factor, lambda_r, epochs, entropy_coef, actor_lr, critic_lr, num_actions,
                 input_dim):
        self.epsilon = epsilon
        self.loss_history = []
        self.discount_factor = discount_factor
        self.lambda_r = lambda_r
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.memory = PPOMemory()

        self.policy = PolicyNet(self.num_actions, self.input_dim).to(device)
        self.value = ValueNet(self.input_dim).to(device)
        self.mse_loss = nn.MSELoss()
        self.p_optim = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        self.c_optim = optim.Adam(self.value.parameters(), lr=self.critic_lr)

    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            distributions = self.policy(state)
            action_dist_1 = torch.distributions.Normal(distributions[0], distributions[2])
            action_dist_2 = torch.distributions.Normal(distributions[1], distributions[3])
            action_1 = action_dist_1.sample()
            action_2 = action_dist_2.sample()
            action_log_1 = action_dist_1.log_prob(action_1)
            action_log_2 = action_dist_2.log_prob(action_2)

        return [action_1, action_2], [action_log_1, action_log_2]

    def evaluate(self, state, action):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        distributions = self.policy(state)
        action_dist_1 = torch.distributions.Normal(distributions[:][0], distributions[:][2])
        action_dist_2 = torch.distributions.Normal(distributions[:][1], distributions[:][3])
        action_log_1 = action_dist_1.log_prob(action[:][0])
        action_log_2 = action_dist_2.log_prob(action[:][1])
        action_logs = torch.cat([action_log_1, action_log_2], dim=1).to(device)
        entropies = torch.cat([action_dist_1.entropy(), action_dist_2.entropy()], dim=1).to(device)
        state_values = self.value(state)

        return action_logs, entropies, state_values

    def calc_GAE_V_tar(self, states, rewards, dones):
        with torch.no_grad():
            advantages = []
            v_targets = []
            state_values = self.value(states)
            global_delta = deque()
            global_delta_v = deque()
            for i in range(len(states)):
                gamma = self.discount_factor
                k_limit = 1
                delta = deque()
                delta_v = deque()
                for j in range(i, len(rewards)):
                    summation = rewards[i]
                    for k in range(i + 1, k_limit + i):
                        summation += gamma * rewards[k]
                        gamma *= self.discount_factor
                    if not dones[k_limit + i - 1]:
                        summation += state_values[k_limit + i]
                    delta_v.append(summation)
                    summation -= state_values[i]
                    delta.append(summation)
                    k_limit += 1

                global_delta.append(delta)
                global_delta_v.append(delta_v)

            for i in range(len(states)):
                landa = self.lambda_r
                summation = global_delta[i][0]
                summation_v = global_delta_v[i][0]
                for j in range(1, len(global_delta[i]) - 1):
                    summation += landa * global_delta[i][j]
                    summation_v += landa * global_delta_v[i][j]
                    landa *= self.lambda_r
                summation = (1 - self.lambda_r) * summation + landa * global_delta[i][j]
                summation_v = (1 - self.lambda_r) * summation_v + landa * global_delta_v[i][j]
                advantages.append(summation)
                v_targets.append(summation_v)

            return torch.as_tensor(advantages, dtype=torch.float32, device=device), torch.as_tensor(v_targets,
                                                                                                    dtype=torch.float32,
                                                                                                    device=device)

    def learn(self):
        states, actions, old_action_logs, rewards, next_states, dones = self.memory.return_samples()
        advantages, v_targets = self.calc_GAE_V_tar(states, rewards, dones)

        for _ in range(self.epochs):
            action_logs, entropies, state_values = self.evaluate(states, actions)
            ratios = torch.exp(action_logs - old_action_logs)
            surrogate_1 = ratios * advantages
            surrogate_2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = (- torch.min(surrogate_1, surrogate_2) - self.entropy_coef * entropies).mean()
            value_loss = self.mse_loss(state_values, v_targets)
            self.loss_history.append(policy_loss + value_loss)

            self.p_optim.zero_grad()
            policy_loss.backward()
            self.p_optim.step()

            self.c_optim.zero_grad()
            value_loss.backward()
            self.c_optim.step()

