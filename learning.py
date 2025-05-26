import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from logger import Logger
from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic
from buffer.episode import Episode
from buffer.ppo.module import BufferPPO
from tqdm import tqdm
import gymnasium
import os


class PPO(Learning):
    def __init__(
        self,
        environment: gymnasium.Env,
        hidden_layers: tuple[int],
        epochs: int,
        buffer_size: int,
        batch_size: int,
        gamma: float,
        gae_lambda,
        policy_clip,
        learning_rate,
        log_path: str = None,
    ) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)
        self.env = environment
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip

        self.buffer = BufferPPO(
            gamma=gamma,
            max_size=buffer_size,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )

        self.logger = Logger(log_path)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.to(self.device)

        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm = 0.0

    def take_action(
        self, state: np.ndarray, action_mask: np.ndarray
    ) -> tuple[int, float, float]:
        state = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.device)
        mask = T.tensor(action_mask, dtype=T.bool).unsqueeze(0).to(self.device)

        dist = self.actor(state, mask)
        action = dist.sample()

        while not mask[0, action].item():
            action = dist.sample()

        logp = dist.log_prob(action).item()
        value = self.critic(state).item()
        idx = action.item()

        source_pos, target_pos, _ = self.env.get_all_actions(self.env.turn)
        from_pos = tuple(source_pos[idx])
        to_pos = tuple(target_pos[idx])
        for name, pos in self.env.pieces[self.env.turn].items():
            if isinstance(pos, tuple) and pos == from_pos:
                self.logger.log(f"{name} {self.env.turn}: {from_pos} -> {to_pos}")
                break

        return idx, logp, value

    def epoch(self):
        actor_grad_norms = []
        critic_grad_norms = []

        data = self.buffer.sample()
        if not data or len(data[0]) == 0:
            self.last_actor_grad_norm = 0.0
            self.last_critic_grad_norm = 0.0
            return

        (
            states_arr,
            actions_arr,
            rewards_arr,
            dones_arr,
            old_probs_arr,
            values_arr,
            masks_arr,
            advantages_arr,
            batches,
        ) = data

        for batch in batches:
            masks = T.tensor(masks_arr[batch], dtype=T.bool).to(self.device)
            states = T.tensor(states_arr[batch], dtype=T.float32).to(self.device)
            values = T.tensor(values_arr[batch], dtype=T.float32).to(self.device)
            actions = T.tensor(actions_arr[batch], dtype=T.long).to(self.device)
            old_probs = T.tensor(old_probs_arr[batch], dtype=T.float32).to(self.device)
            advantages = T.tensor(advantages_arr[batch], dtype=T.float32).to(
                self.device
            )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dist = self.actor(states, masks)
            entropy = dist.entropy().mean()
            critic_value = self.critic(states).squeeze()
            new_probs = dist.log_prob(actions)

            prob_ratio = (new_probs - old_probs).exp()
            weighted_probs = advantages * prob_ratio
            clipped_probs = (
                T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages
            )

            actor_loss = -T.min(weighted_probs, clipped_probs).mean()
            actor_loss -= self.entropy_coef * entropy

            returns = advantages + values
            critic_loss = F.mse_loss(critic_value, returns)

            total_loss = actor_loss + self.value_coef * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            actor_sq = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.actor.parameters()
                if p.grad is not None
            )
            critic_sq = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.critic.parameters()
                if p.grad is not None
            )
            actor_grad_norms.append(T.sqrt(T.tensor(actor_sq)).item())
            critic_grad_norms.append(T.sqrt(T.tensor(critic_sq)).item())

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.last_actor_grad_norm = (
            float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0
        )
        self.last_critic_grad_norm = (
            float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0
        )

    def learn(self):
        for _ in tqdm(
            range(self.epochs), desc="PPO Learning...", ncols=64, leave=False
        ):
            self.epoch()

    def remember(self, episode: Episode):
        if len(episode.states) > 0:
            self.buffer.add(episode)

    def save(self, folder: str, name: str) -> None:
        path = os.path.join(folder, f"{name}.pt")
        T.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, folder: str, name: str):
        path = os.path.join(folder, f"{name}.pt")
        if not os.path.exists(path):
            self.logger.log(
                f"Warning: Model file {path} does not exist. Using random init."
            )
            return
        ckpt = T.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.actor.to(self.device)
        self.critic.to(self.device)
