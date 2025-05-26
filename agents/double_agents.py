from copy import deepcopy
import os
import numpy as np

from agents.base import BaseAgent
from buffer.episode import Episode
from learnings.base import Learning


class DoubleAgentsIntellector(BaseAgent):
    def __init__(
        self,
        env,
        learner: Learning,
        episodes: int,
        train_on: int,
        result_folder: str,
    ) -> None:
        super().__init__(env, learner, episodes, train_on, result_folder)
        self.white_agent = deepcopy(learner)
        self.black_agent = deepcopy(learner)

        self.white_agent.buffer.clear()
        self.black_agent.buffer.clear()

    def take_action(self, turn: int, episode: Episode):
        mask = self.env.get_all_actions(turn)[-1]
        state = self.env.get_state(turn)

        current_agent = self.white_agent if turn == 0 else self.black_agent
        action, log_prob, value = current_agent.take_action(state, mask)

        next_state, total_rewards, done, truncated, info = self.env.step(action)

        current_agent.logger.log(
            f"Ep {self.current_ep + 1} | Turn {turn} | "
            f"Action {action} | Reward {total_rewards[turn]:.4f} | Done {done}"
        )

        episode.add(state, total_rewards[turn], action, done, log_prob, value, mask)
        return done, next_state, total_rewards, action, log_prob, value, mask, info

    def add_episodes(self, white: Episode, black: Episode) -> None:
        self.white_agent.remember(white)
        self.black_agent.remember(black)

    def learn(self) -> None:
        if len(self.white_agent.buffer) > 0:
            self.white_agent.learn()
            actor_norm = self.white_agent.last_actor_grad_norm
            critic_norm = self.white_agent.last_critic_grad_norm

            if np.isnan(actor_norm) or np.isinf(actor_norm):
                actor_norm = 0.0

            if np.isnan(critic_norm) or np.isinf(critic_norm):
                critic_norm = 0.0

            self.grad_norms["white_actor"].append(actor_norm)
            self.grad_norms["white_critic"].append(critic_norm)

        if len(self.black_agent.buffer) > 0:
            self.black_agent.learn()
            actor_norm = self.black_agent.last_actor_grad_norm
            critic_norm = self.black_agent.last_critic_grad_norm

            if np.isnan(actor_norm) or np.isinf(actor_norm):
                actor_norm = 0.0

            if np.isnan(critic_norm) or np.isinf(critic_norm):
                critic_norm = 0.0

            self.grad_norms["black_actor"].append(actor_norm)
            self.grad_norms["black_critic"].append(critic_norm)

    def save_learners(self) -> None:
        self.white_agent.save(self.result_folder, "white_ppo")
        self.black_agent.save(self.result_folder, "black_ppo")

    def load_learners(self) -> None:
        white_path = os.path.join(self.result_folder, "white_ppo.pt")
        black_path = os.path.join(self.result_folder, "black_ppo.pt")

        if os.path.exists(white_path):
            self.white_agent.load(self.result_folder, "white_ppo")

        if os.path.exists(black_path):
            self.black_agent.load(self.result_folder, "black_ppo")
