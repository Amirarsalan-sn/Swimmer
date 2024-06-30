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
        # rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        # dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, action_logs, self.rewards, next_states, self.dones

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
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_actions),
            nn.Tanh()
        )

        """self.mean = nn.Sequential(
            nn.Linear(16, num_actions)
        )"""

        """self.std = nn.Sequential(
            nn.Linear(16, num_actions),
            nn.ReLU(inplace=True)
        )"""

        # He initialization.
        for layer in [self.FN]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

        """for layer in [self.mean]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')"""

        """ for layer in [self.std]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')"""

    def forward(self, x):
        """if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        mid_x = self.FN(x)
        mean = self.mean(mid_x)
        std = self.std(mid_x)

        return torch.cat([mean, std], dim=1).to(device)"""
        return self.FN(x)


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
                 actor_lr, critic_lr, num_actions,
                 input_dim):
        self.epsilon = epsilon
        self.loss_history = []
        self.initial_std = initial_std
        self.std_coef = std_coef
        self.std_min = std_min
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
            action_dist_1 = torch.distributions.Normal(distributions[0], self.initial_std)
            action_dist_2 = torch.distributions.Normal(distributions[1], self.initial_std)
            action_1 = action_dist_1.sample()
            action_2 = action_dist_2.sample()
            action_log_1 = action_dist_1.log_prob(action_1)
            action_log_2 = action_dist_2.log_prob(action_2)

        return [action_1.item(), action_2.item()], [action_log_1.item(), action_log_2.item()]

    def evaluate(self, state, action):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        distributions = self.policy(state)
        action_dist_1 = torch.distributions.Normal(distributions[:, 0], self.initial_std)
        action_dist_2 = torch.distributions.Normal(distributions[:, 1], self.initial_std)
        action_log_1 = action_dist_1.log_prob(action[:, 0])
        action_log_2 = action_dist_2.log_prob(action[:, 1])
        action_log_1 = action_log_1.unsqueeze(1)
        action_log_2 = action_log_2.unsqueeze(1)

        action_logs = torch.cat([action_log_1, action_log_2], dim=1).to(device)
        entropies = torch.cat([action_dist_1.entropy().unsqueeze(1), action_dist_2.entropy().unsqueeze(1)], dim=1).to(
            device)
        state_values = self.value(state)

        return action_logs, entropies, state_values

    def decay_std(self):
        self.initial_std = max(self.std_min, self.std_coef * self.initial_std)
        return self.initial_std

    def calc_GAE_V_tar(self, states, rewards, dones):
        with torch.no_grad():
            advantages = []
            v_targets = []
            state_values = self.value(states)
            delta = deque()
            for i in range(len(rewards)):
                if not dones[i]:
                    delta.append(
                        rewards[i] + self.discount_factor * state_values[i + 1].item() - state_values[i].item())
                else:
                    delta.append(rewards[i] - state_values[i].item())

            summation = 0
            for i in range(len(rewards) - 1, -1, -1):
                summation = self.discount_factor * self.lambda_r * summation + delta[i]
                advantages.insert(0, summation)
                v_targets.insert(0, summation + state_values[i].item())

            return torch.as_tensor(advantages, dtype=torch.float32, device=device), torch.as_tensor(v_targets,
                                                                                                    dtype=torch.float32,
                                                                                                    device=device)

    def learn(self):
        states, actions, old_action_logs, rewards, next_states, dones = self.memory.return_samples()
        advantages, v_targets = self.calc_GAE_V_tar(states, rewards, dones)
        advantages = advantages.unsqueeze(1)
        v_targets = v_targets.unsqueeze(1)
        for _ in range(self.epochs):
            action_logs, entropies, state_values = self.evaluate(states, actions)
            ratios = torch.exp(action_logs - old_action_logs)
            surrogate_1 = ratios * advantages
            surrogate_2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = (- torch.min(surrogate_1, surrogate_2) - self.entropy_coef * entropies).mean()
            value_loss = self.mse_loss(state_values, v_targets)

            self.p_optim.zero_grad()
            policy_loss.backward()
            self.p_optim.step()

            self.c_optim.zero_grad()
            value_loss.backward()
            self.c_optim.step()

            self.loss_history.append(policy_loss.item() + value_loss.item())

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
        state[0] = (state[0] + 50) / 100
        state[1] = (state[1] + 50) / 100
        state[2] = (state[2] + 50) / 100
        state[3] = (state[3] + 50) / 100
        state[4] = (state[4] + 50) / 100
        state[5] = (state[5] + 50) / 100
        state[6] = (state[6] + 50) / 100
        state[7] = (state[7] + 50) / 100

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
        self.clip_or_kl = hyperparams["clip_or_kl"]
        self.lambda_r = hyperparams["lambda_r"]
        self.epochs = hyperparams["epochs"]
        self.epsilon = hyperparams["epsilon"]
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
        if self.clip_or_kl:
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
                                 self.env.observation_space.shape[0])
        else:
            pass  # PPO KL

    def train(self):
        total_steps = 0
        self.reward_history = []
        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            if episode == 99:
                a = 1
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            while not done and not truncation:
                actions, action_logs = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(actions)
                self.agent.memory.store(state, actions, action_logs, reward, next_state, (done or truncation))
                state = next_state
                episode_reward += reward
                total_steps += 1
                step_size += 1

            self.agent.learn()
            self.agent.memory.delete_mem()
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
                      f"STD: {self.agent.initial_std: .2f} "
                      f"Loss: {self.agent.loss_history[-1]:.2f}")

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
                # print(f'y speed: {state[3]}, y pos: {state[1]}, reward : {reward}')
                action, log_action = self.agent.select_action(state)
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
                plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Error in sma:\n{e}')

        try:
            plt.figure()
            plt.title("Network Loss")
            plt.plot(self.agent.loss_history, label='Loss', color='#8921BB', alpha=1)
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.tight_layout()

            # Only save as file if last episode
            if episode == self.max_episodes:
                plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Error in loss:\n{e}')


if __name__ == "__main__":
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": './clip/final_weights' + '_' + '400' + '.pth',
        "save_path": './clip/final_weights',
        "save_interval": 100,

        "actor_learning_rate": 75e-5,
        "critic_learning_rate": 75e-5,
        "discount_factor": 0.99,
        "entropy_coef": 0.003,
        "initial_std": 0.2 if train_mode else 0.01,
        "std_min": 0.01,
        "std_coef": 0.99,
        "lambda_r": 0.95,
        "epsilon": 0.2,
        "epochs": 10,
        "max_episodes": 1000 if train_mode else 2,
        "render": render,

        "clip_or_kl": True,

        "render_fps": 60,
    }

    # Run
    DRL = Agent(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(2)
