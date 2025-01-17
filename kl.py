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


class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """

        self.capacity = capacity

        self.states = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """

        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """

        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(
            device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, next_states, rewards, dones

    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque
        represents the length of the entire memory.
        """

        return len(self.dones)


class PPOMemory:
    def __init__(self, capacity):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.action_logs = deque(maxlen=capacity)
        self.means_stds = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, action_log, mean_stds, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_logs.append(action_log)
        self.means_stds.append(mean_stds)
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
        mean_stds = torch.stack(
            [torch.as_tensor(self.means_stds[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        # rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        # dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, action_logs, mean_stds, self.rewards, next_states, self.dones

    def delete_mem(self):
        self.states.clear()
        self.actions.clear()
        self.action_logs.clear()
        self.means_stds.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.dones)


class PolicyNet(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(PolicyNet, self).__init__()

        self.FN = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        self.mean = nn.Sequential(
            nn.Linear(64, num_actions),
            nn.Tanh()
        )

        self.std = nn.Sequential(
            nn.Linear(64, num_actions),
            nn.Sigmoid()
        )

        # He initialization.
        """for layer in [self.FN]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

        for layer in [self.mean]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='tanh')"""

        """for layer in [self.std]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='softmax')"""

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        mid_x = self.FN(x)
        mean = self.mean(mid_x)
        std = self.std(mid_x)

        return torch.cat([mean, std], dim=1).to(device)
        # return self.FN(x)


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
    def __init__(self, epsilon, discount_factor, lambda_r, epochs, entropy_coef, initial_std, std_coef, std_min,
                 actor_lr, critic_lr, num_actions, input_dim, replay_capacity, batch_size, clip_gradient_norm,
                 clip_or_kl=True, kl_coef=0, target_kl=0):
        self.epsilon = epsilon
        self.policy_loss_history = []
        self.policy_running_loss = 0
        self.policy_learned_count = 0
        self.value_loss_history = []
        self.value_running_loss = 0
        self.value_learned_count = 0
        self.initial_std = torch.full((num_actions,), initial_std, dtype=torch.float32).to(device)
        self.std_coef = std_coef
        self.std_min = torch.full((num_actions,), std_min, dtype=torch.float32).to(device)
        self.discount_factor = discount_factor
        self.lambda_r = lambda_r
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.clip_gradient_norm = clip_gradient_norm
        self.replay_memory = ReplayMemory(capacity=replay_capacity)
        self.memory = PPOMemory(capacity=replay_capacity)
        self.clip_or_kl = clip_or_kl

        self.policy = PolicyNet(self.num_actions, self.input_dim).to(device)
        self.value = ValueNet(self.input_dim).to(device)
        self.mse_loss = nn.MSELoss()
        self.p_optim = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        self.c_optim = optim.Adam(self.value.parameters(), lr=self.critic_lr)

    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            means_stds = self.policy(state)[0]
            means = means_stds[0:2]
            stds = means_stds[2:]
            covariance_matrix = torch.diag(stds).unsqueeze(dim=0)
            distribution = torch.distributions.MultivariateNormal(means, covariance_matrix)
            action = distribution.sample()
            action_log = distribution.log_prob(action)

        return action.tolist()[0], action_log.item(), means_stds

    def kl_divergence(self, state, action, old_mean_stds):
        mean_stds = self.policy(state)
        means = mean_stds[:, 0:2]
        stds = mean_stds[:, 2:]
        covariance_matrices = torch.diag_embed(stds).to(device)
        old_means = old_mean_stds[:, 0:2]
        old_stds = old_mean_stds[:, 2:]
        old_covariance_matrices = torch.diag_embed(old_stds).to(device)
        distributions = torch.distributions.MultivariateNormal(means, covariance_matrices)
        old_distributions = torch.distributions.MultivariateNormal(old_means, old_covariance_matrices)
        action_logs = distributions.log_prob(action)
        kl = torch.distributions.kl_divergence(distributions, old_distributions)
        return action_logs, kl

    def evaluate(self, state, action):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        means_stds = self.policy(state)
        means = means_stds[:, 0:2]
        stds = means_stds[:, 2:]
        covariance_matrices = torch.diag_embed(stds).to(device)
        distributions = torch.distributions.MultivariateNormal(means, covariance_matrices)
        action_logs = distributions.log_prob(action)
        entropies = distributions.entropy()

        return action_logs, entropies

    def decay_std(self):
        self.initial_std = torch.max(self.std_min, self.std_coef * self.initial_std)
        return self.initial_std

    def calc_GAE_V_tar(self, states, next_state, rewards, dones):
        with torch.no_grad():
            advantages = []
            state_values = self.value(states)
            next_state_value = self.value(next_state)
            delta = deque()
            for i in range(len(rewards)):
                if (not dones[i]) and (i < (len(rewards) - 1)):
                    delta.append(
                        rewards[i] + self.discount_factor * state_values[i + 1].item() - state_values[i].item())
                elif (not dones[i]) and (i == (len(rewards) - 1)):
                    delta.append(
                        rewards[i] + self.discount_factor * next_state_value.item() - state_values[i].item())
                else:
                    delta.append(rewards[i] - state_values[i].item())

            summation = 0
            for i in range(len(rewards) - 1, -1, -1):
                summation = self.discount_factor * self.lambda_r * summation + delta[i]
                advantages.insert(0, summation)

            return torch.as_tensor(advantages, dtype=torch.float32, device=device)

    def learn_policy(self, done):
        if self.clip_or_kl:
            self.learn_policy_clip(done)
        else:
            self.learn_policy_kl(done)

    def learn_policy_clip(self, done):
        states, actions, old_action_logs, means_std, rewards, next_states, dones = self.memory.return_samples()
        advantages = self.calc_GAE_V_tar(states, next_states[-1], rewards, dones)
        del means_std
        # advantages = advantages.unsqueeze(1)
        for _ in range(self.epochs):
            action_logs, entropies = self.evaluate(states, actions)
            ratios = torch.exp(action_logs - old_action_logs)
            surrogate_1 = ratios * advantages
            surrogate_2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = (- torch.min(surrogate_1, surrogate_2) - self.entropy_coef * entropies).mean()

            self.policy_running_loss += policy_loss.item()
            self.policy_learned_count += 1

            self.p_optim.zero_grad()
            policy_loss.backward()
            self.p_optim.step()
        if done:
            self.policy_loss_history.append(self.policy_running_loss / self.policy_learned_count)
            self.policy_running_loss = 0
            self.policy_learned_count = 0

    def learn_policy_kl(self, done):
        states, actions, old_action_logs, means_std, rewards, next_states, dones = self.memory.return_samples()
        advantages = self.calc_GAE_V_tar(states, next_states[-1], rewards, dones)
        # advantages = advantages.unsqueeze(1)
        for _ in range(self.epochs):
            action_logs, kl = self.kl_divergence(states, actions, means_std)
            ratios = torch.exp(action_logs - old_action_logs)
            surrogate = ratios * advantages
            kl_divergence = self.kl_coef * kl
            policy_loss = - (surrogate - kl_divergence).mean()
            self.policy_running_loss += policy_loss.item()
            self.policy_learned_count += 1

            self.p_optim.zero_grad()
            policy_loss.backward()
            self.p_optim.step()

            kl_mean = kl.mean()
            if kl_mean < self.target_kl / 1.5:
                self.kl_coef /= 2
            elif kl_mean > self.target_kl * 1.5:
                self.kl_coef *= 2
        if done:
            self.policy_loss_history.append(self.policy_running_loss / self.policy_learned_count)
            self.policy_running_loss = 0
            self.policy_learned_count = 0

    def learn_v(self, done):
        states, next_states, rewards, dones = self.replay_memory.sample(self.batch_size)
        v_preds = self.value(states)
        with torch.no_grad():
            next_values = self.value(next_states)
        next_values[dones] = 0
        v_tar = rewards + self.discount_factor * next_values
        loss = self.mse_loss(v_preds, v_tar)

        self.value_running_loss += loss.item()
        self.value_learned_count += 1

        if done:
            self.value_loss_history.append(self.value_running_loss / self.value_learned_count)
            self.value_running_loss = 0
            self.value_learned_count = 0

        self.c_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.value.parameters(), self.clip_gradient_norm)
        self.c_optim.step()

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.policy.state_dict(), path)


class StepWrapper(gym.Wrapper):
    """
    A wrapper class for modifying the state and reward functions of the
    Swimmer v4 environment.
    """

    def __init__(self, env):
        """
        Initializes the StepWrapper. This is the main class for wrapping the environment with it.

        Args:
            env (gym.Env): The environment to be wrapped.

        Attributes:
            reward_wrapper (Inherit from RewardWrapper):
                An instance of the RewardWrapper class for modifying rewards.
        """
        super().__init__(env)  # We give the env here to initialize the gym.Wrapper superclass (inherited).
        self.observation_wrapper = ObservationWrapper(env)
        self.reward_wrapper = RewardWrapper(env)

    def step(self, action: list):
        """
        Executes a step in the environment with the provided action.The reason
        behind using this method is to have access to the state and reward functions return.

        Args:
            action (int): The action to be taken.
        """

        state, reward, done, truncation, info = self.env.step(action)

        modified_state = self.observation_wrapper.observation(state)
        # modified_reward = self.reward_wrapper.reward(modified_state)
        return modified_state, reward, done, truncation, info

    def reset(self, seed):
        state, info = self.env.reset(seed=seed)  # Same as before as usual
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info  # Same as before as usual but with returning the modified version of the state


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, state):  # state normalizer
        state = np.array(state)
        state[0] = (state[0] + 5) / 10
        state[1] = (state[1] + 5) / 10
        state[2] = (state[2] + 5) / 10
        state[3] = (state[3] + 5) / 10
        state[4] = (state[4] + 5) / 10
        state[5] = (state[5] + 5) / 10
        state[6] = (state[6] + 5) / 10
        state[7] = (state[7] + 5) / 10

        return state


class RewardWrapper(gym.RewardWrapper):
    """
    Wrapper class for modifying rewards in the MountainCar-v0 environment.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, state):
        """
        Modifies the reward based on the current state of the environment.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            float: The modified reward.
        """
        return 0


class Agent:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.actor_learning_rate = hyperparams["actor_learning_rate"]
        self.critic_learning_rate = hyperparams["critic_learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.entropy_coef = hyperparams["entropy_coef"]
        self.initial_std = hyperparams["initial_std"]
        self.std_min = hyperparams["std_min"]
        self.std_coef = hyperparams["std_coef"]
        self.kl_coef = hyperparams["kl_coef"]
        self.target_kl = hyperparams["target_kl"]
        self.clip_or_kl = hyperparams["clip_or_kl"]
        self.lambda_r = hyperparams["lambda_r"]
        self.epochs = hyperparams["epochs"]
        self.epsilon = hyperparams["epsilon"]
        self.replay_capacity = hyperparams["replay_capacity"]
        self.batch_size = hyperparams["batch_size"]
        self.clip_gradient_norm = hyperparams["clip_gradient_norm"]
        self.max_episodes = hyperparams["max_episodes"]
        self.render = hyperparams["render"]
        self.render_fps = hyperparams["render_fps"]
        self.reward_history = None

        # Define Env
        self.env = gym.make('Swimmer-v4', render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps  # For max frame rate make it 0
        warnings.filterwarnings("ignore", category=UserWarning)

        # Apply RewardWrapper
        self.env = StepWrapper(self.env)

        self.agent = PPOClip(self.epsilon,
                             self.discount_factor,
                             self.lambda_r,
                             self.epochs,
                             self.entropy_coef,
                             self.initial_std,
                             self.std_coef,
                             self.std_min,
                             self.actor_learning_rate,
                             self.critic_learning_rate,
                             self.env.action_space.shape[0],
                             self.env.observation_space.shape[0],
                             self.replay_capacity,
                             self.batch_size,
                             self.clip_gradient_norm,
                             self.clip_or_kl,
                             self.kl_coef,
                             self.target_kl)

    def train(self):
        total_steps = 0
        self.reward_history = []
        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            while not done and not truncation:
                actions, action_logs, mean_stds = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(actions)
                self.agent.memory.store(state, actions, action_logs, mean_stds, reward, next_state,
                                        (done or truncation))
                self.agent.replay_memory.store(state, next_state, reward, done)

                if len(self.agent.memory) == self.batch_size or (done or truncation):
                    self.agent.learn_v(done or truncation)
                    self.agent.learn_policy(done or truncation)
                    self.agent.memory.delete_mem()

                state = next_state
                episode_reward += reward
                total_steps += 1
                step_size += 1

            self.agent.decay_std()
            self.reward_history.append(episode_reward)

            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"STD: {self.agent.initial_std[0].item(): .2f}, "
                      f"Policy Loss: {self.agent.policy_loss_history[-1]:.2f}, "
                      f"Value Loss: {self.agent.value_loss_history[-1]:.2f}")

            print(result)
        self.plot_training(episode)

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.policy.load_state_dict(torch.load(self.RL_load_path))
        self.agent.policy.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            reward = 0
            while not done and not truncation:
                action, log_action, mean_std = self.agent.select_action(state)
                print('-----------------------------------------------------')
                print(f'{state}')
                print(f'{action}')
                print('-----------------------------------------------------')
                next_state, reward, done, truncation, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()  # close the rendering window

    def plot_training(self, episode):
        try:
            # Calculate the Simple Moving Average (SMA) with a window size of 50
            sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

            # Clip max (high) values for better plot analysis
            reward_history = np.clip(self.reward_history, a_min=-100, a_max=100)
            sma = np.clip(sma, a_min=None, a_max=100)

            plt.figure()
            plt.title("Obtained Rewards")
            plt.plot(reward_history, label='Raw Reward', color='#4BA754', alpha=1)
            plt.plot(sma, label='SMA 50', color='#F08100')
            plt.xlabel("Episode")
            plt.ylabel("Rewards")
            plt.legend()
            plt.tight_layout()

            # Only save as file if last episode
            if episode == self.max_episodes:
                plt.savefig('./images/reward_plot_kl.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Error in sma:\n{e}')

        try:
            plt.figure()
            plt.title("Policy Loss")
            plt.plot(self.agent.policy_loss_history, label='Loss', color='#8921BB', alpha=1)
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.tight_layout()

            # Only save as file if last episode
            if episode == self.max_episodes:
                plt.savefig('./images/Policy_Loss_plot_kl.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Error in policy loss:\n{e}')

        try:
            plt.figure()
            plt.title("Value Loss")
            plt.plot(self.agent.value_loss_history, label='Loss', color='#8921BB', alpha=1)
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.tight_layout()

            # Only save as file if last episode
            if episode == self.max_episodes:
                plt.savefig('./images/Value_Loss_plot_kl.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Error in value loss:\n{e}')


if __name__ == "__main__":
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": './kl/final_weights' + '_' + '100' + '.pth',
        "save_path": './kl/final_weights',
        "save_interval": 100,

        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "discount_factor": 0.99,
        "entropy_coef": 0.003,
        "initial_std": 0.2 if train_mode else 0.001,
        "std_min": 0.001,
        "std_coef": 0.99,
        "lambda_r": 0.95,
        "epsilon": 0.2,
        "epochs": 10,
        "replay_capacity": 125_000,
        "kl_coef": 0.5,
        "target_kl": 0.2,
        "batch_size": 64,
        "clip_gradient_norm": 5,
        "max_episodes": 1000 if train_mode else 2,
        "render": render,

        "clip_or_kl": False,

        "render_fps": 60,
    }

    # Run
    DRL = Agent(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(1)
