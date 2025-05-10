import os
import gymnasium
import numpy as np
import torch as T
import torch.optim as optim

from tqdm import tqdm
from buffer.ppo import BufferPPO
from buffer.episode import Episode
from learning import Logger

from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic


class PPO(Learning):
    def __init__(
            self,
            environment: gymnasium.Env,
            hidden_layers: tuple[int],
            epochs: int,
            buffer_size: int,
            batch_size: int,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            policy_clip: float = 0.2,
            learning_rate: float = 0.003,
            log_path: str = None,
    ) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)

        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm = 0.0
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer = BufferPPO(
            gamma=gamma,
            max_size=buffer_size,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )

        self.logger = Logger(log_path)
        self.hidden_layers = hidden_layers
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.to(self.device)

    def take_action(self, state: np.ndarray, action_mask: np.ndarray):
        state = T.Tensor(state).unsqueeze(0).to(self.device)
        action_mask = T.Tensor(action_mask).unsqueeze(0).to(self.device)
        dist = self.actor(state, action_mask)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(self.critic(state)).item()
        action = T.squeeze(action).item()

        source_pos, target_pos, action_mask_np = self.env.get_all_actions(self.env.turn)
        from_pos = tuple(source_pos[action])
        to_pos = tuple(target_pos[action])

        for name, pos in self.env.pieces[self.env.turn].items():
            if isinstance(pos, tuple) and pos == from_pos:
                self.logger.log(f"{name} {self.env.turn}: {from_pos} -> {to_pos}")
                break

        return action, probs, value

    def epoch(self):
        actor_grad_norms = []
        critic_grad_norms = []
        (
            states_arr,
            actions_arr,
            rewards_arr,
            goals_arr,
            old_probs_arr,
            values_arr,
            masks_arr,
            advantages_arr,
            batches,
        ) = self.buffer.sample()

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
            weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantages
            )

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            critic_loss = ((advantages + values - critic_value) ** 2).mean()
            total_loss = actor_loss + 0.5 * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            actor_norm_sq = 0.0
            for p in self.actor.parameters():
                if p.grad is not None:
                    actor_norm_sq += p.grad.data.norm() ** 2
            critic_norm_sq = 0.0
            for p in self.critic.parameters():
                if p.grad is not None:
                    critic_norm_sq += p.grad.data.norm() ** 2

            actor_grad_norms.append(T.sqrt(actor_norm_sq).item())
            critic_grad_norms.append(T.sqrt(critic_norm_sq).item())

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def learn(self):
        for epoch in tqdm(range(self.epochs), desc="PPO Learning...", ncols=64, leave=False):
            self.epoch()
        self.buffer.clear()

    def remember(self, episode: Episode):
        self.buffer.add(episode)

    def save(self, folder: str, name: str):
        T.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            os.path.join(folder, f"{name}.pt"),
        )

    def load(self, folder: str, name: str):
        checkpoint = T.load(
            os.path.join(folder, f"{name}.pt"), map_location=self.device
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor.to(self.device)
        self.critic.to(self.device)
