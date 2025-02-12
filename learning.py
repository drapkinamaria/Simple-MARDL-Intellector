import numpy as np
import torch as T
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import gymnasium
import os

def build_base_model(input_size: int, hidden_layers: tuple[int], output_size: int, last_activation: nn.Module = nn.Identity()) -> nn.Module:
    layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
    for i in range(len(hidden_layers) - 1):
        layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.ReLU()]
    layers += [nn.Linear(hidden_layers[-1], output_size), last_activation]
    return nn.Sequential(*layers)

def make_batch_ids(n: int, batch_size: int, shuffle: bool = True) -> np.ndarray:
    starts = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    if shuffle:
        np.random.shuffle(indices)
    return [indices[i : i + batch_size] for i in starts]

def tensor_to_numpy(x: T.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.base_model = build_base_model(state_dim, hidden_layers, action_dim, nn.Softmax(dim=1))

    def forward(self, states: T.Tensor, action_mask: T.Tensor):
        x = self.base_model(states)
        s = action_mask.sum(dim=1)
        l = ((x * (1 - action_mask)).sum(dim=1) / s).unsqueeze(1)
        x = (x + l) * action_mask
        return Categorical(x)

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.model = build_base_model(state_dim, hidden_layers, 1)

    def forward(self, state: T.Tensor):
        return self.model(state)

# Определение базового класса Learning
class Learning(nn.Module, ABC):
    def __init__(self, environment: gymnasium.Env, epochs: int, gamma: float, learning_rate: float) -> None:
        super().__init__()
        self.state_dim = np.prod(environment.observation_space.shape)
        self.action_dim = environment.action_space.n
        self.gamma = gamma
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    @abstractmethod
    def take_action(self, state: np.ndarray, *args):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, *args):
        pass

    @abstractmethod
    def save(self, folder: str, name: str):
        pass

# Определение класса Episode
class Episode:
    def __init__(self):
        self.goals = []
        self.probs = []
        self.masks = []
        self.values = []
        self.states = []
        self.rewards = []
        self.actions = []

    def add(self, state: np.ndarray, reward: float, action, goal: bool, prob: float = None, value: float = None, masks: np.ndarray = None):
        self.goals.append(goal)
        self.states.append(state)
        self.rewards.append(float(reward))
        self.actions.append(action)
        if prob is not None:
            self.probs.append(prob)
        if value is not None:
            self.values.append(value)
        if masks is not None:
            self.masks.append(masks)

    def calc_advantage(self, gamma: float, gae_lambda: float) -> np.ndarray:
        n = len(self.rewards)
        advantages = np.zeros(n)
        for t in range(n - 1):
            discount = 1
            for k in range(t, n - 1):
                advantages[t] += (discount * (self.rewards[k] + gamma * self.values[k + 1] * (1 - int(self.goals[k]))) - self.values[k])
                discount *= gamma * gae_lambda
        return list(advantages)

    def __len__(self):
        return len(self.goals)

    def total_reward(self) -> float:
        return sum(self.rewards)

class PPO(Learning):
    def __init__(self, environment: gymnasium.Env, hidden_layers: tuple[int], epochs: int, buffer_size: int, batch_size: int, gamma: float = 0.99, gae_lambda: float = 0.95, policy_clip: float = 0.2, learning_rate: float = 0.003) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer = BufferPPO(gamma=gamma, max_size=buffer_size, batch_size=batch_size, gae_lambda=gae_lambda)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.to(self.device)

    def take_action(self, state: np.ndarray, action_mask: np.ndarray):
        state = T.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask = T.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        dist = self.actor(state, action_mask)
        action = dist.sample()

        while not action_mask[0, action].item():
            action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(self.critic(state)).item()
        action = T.squeeze(action).item()
        return action, probs, value

    def epoch(self):
        (states_arr, actions_arr, rewards_arr, goals_arr, old_probs_arr, values_arr, masks_arr, advantages_arr, batches) = self.buffer.sample()
        for batch in batches:
            masks = T.Tensor(masks_arr[batch]).to(self.device)
            values = T.Tensor(values_arr[batch]).to(self.device)
            states = T.Tensor(states_arr[batch]).to(self.device)
            actions = T.Tensor(actions_arr[batch]).to(self.device)
            old_probs = T.Tensor(old_probs_arr[batch]).to(self.device)
            advantages = T.Tensor(advantages_arr[batch]).to(self.device)
            dist = self.actor(states, masks)
            critic_value = T.squeeze(self.critic(states))
            new_probs = dist.log_prob(actions)
            prob_ratio = (new_probs - old_probs).exp()
            weighted_probs = advantages * prob_ratio
            weighted_clipped_probs = (T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages)
            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            critic_loss = ((advantages + values - critic_value) ** 2).mean()
            total_loss = actor_loss + 0.5 * critic_loss
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def learn(self):
        for epoch in tqdm(range(self.epochs), desc="PPO Learning...", ncols=64, leave=False):
            self.epoch()
        self.buffer.clear()

    def remember(self, episode: Episode):
        self.buffer.add(episode)

    def save(self, folder: str, name: str):
        T.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, os.path.join(folder, f"{name}.pt"))

    def load(self, folder: str, name: str):
        checkpoint = T.load(os.path.join(folder, f"{name}.pt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor.to(self.device)
        self.critic.to(self.device)


class BufferPPO:
    def __init__(self, gamma: float, max_size: int, batch_size: int, gae_lambda: float):
        self.gamma = gamma
        self.max_size = max_size
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self.buffer = []

    def add(self, episode: Episode):
        self.buffer.append(episode)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self):
        states, actions, rewards, goals, old_probs, values, masks, advantages = [], [], [], [], [], [], [], []
        for episode in self.buffer:
            states += episode.states
            actions += episode.actions
            rewards += episode.rewards
            goals += episode.goals
            old_probs += episode.probs
            values += episode.values
            masks += episode.masks
            advantages += episode.calc_advantage(self.gamma, self.gae_lambda)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        goals = np.array(goals)
        old_probs = np.array(old_probs)
        values = np.array(values)
        masks = np.array(masks)
        advantages = np.array(advantages)
        batches = make_batch_ids(len(states), self.batch_size)
        return states, actions, rewards, goals, old_probs, values, masks, advantages, batches