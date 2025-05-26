from abc import ABC, abstractmethod
import os
import csv
import numpy as np
import tqdm

from learnings.base import Learning
from buffer.episode import Episode


class BaseAgent(ABC):
    def __init__(
        self,
        env,
        learner: Learning,
        episodes: int,
        train_on: int,
        result_folder: str,
    ) -> None:
        self.env = env
        self.learner = learner
        self.episodes = episodes
        self.train_on = train_on
        self.result_folder = result_folder
        self.episodes_since_train = 0
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(os.path.join(self.result_folder, "renders"), exist_ok=True)

        self.current_ep = 0
        self.grad_norms = {
            "white_actor": [],
            "white_critic": [],
            "black_actor": [],
            "black_critic": [],
        }

        self.log_path = os.path.join(self.result_folder, "episode_logs.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "episode",
                        "winner",
                        "reward_black",
                        "reward_white",
                        "steps_to_finish",
                        "actor_grad_norm_black",
                        "critic_grad_norm_black",
                        "actor_grad_norm_white",
                        "critic_grad_norm_white",
                    ]
                )

    def take_action(self, turn: int, episode: Episode):
        mask = self.env.get_all_actions(turn)[-1]
        state = self.env.get_state(turn)
        action, log_prob, value = self.learner.take_action(state, mask)
        next_state, total_rewards, done, truncated, info = self.env.step(action)

        self.learner.logger.log(
            f"Ep {self.current_ep + 1} | Turn {turn} | "
            f"Action {action} | Reward {total_rewards[turn]:.4f} | Done {done}"
        )

        episode.add(state, total_rewards[turn], action, done, log_prob, value, mask)
        return done, next_state, total_rewards, action, log_prob, value, mask, info

    def train_episode(self, render: bool):
        ep_w = Episode()
        ep_b = Episode()
        self.env.reset()
        turn = 0
        total_rewards_white = 0.0
        total_rewards_black = 0.0

        while True:
            done, next_state, total_rewards, action, log_prob, value, mask, info = (
                self.take_action(turn, ep_w if turn == 0 else ep_b)
            )

            total_rewards_white += total_rewards[0]
            total_rewards_black += total_rewards[1]

            if done:
                break
            turn = 1 - turn

        self.add_episodes(ep_w, ep_b)
        self.episodes_since_train += 1

        if self.episodes_since_train >= self.train_on:
            self.learn()
            self.white_agent.buffer.clear()
            self.black_agent.buffer.clear()
            self.episodes_since_train = 0
            self.save_learners()

        agn_b = (
            np.mean(self.grad_norms["black_actor"])
            if self.grad_norms["black_actor"]
            else 0.0
        )
        cgn_b = (
            np.mean(self.grad_norms["black_critic"])
            if self.grad_norms["black_critic"]
            else 0.0
        )
        agn_w = (
            np.mean(self.grad_norms["white_actor"])
            if self.grad_norms["white_actor"]
            else 0.0
        )
        cgn_w = (
            np.mean(self.grad_norms["white_critic"])
            if self.grad_norms["white_critic"]
            else 0.0
        )

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.current_ep + 1,
                    info.get("winner", "draw"),
                    total_rewards_black,
                    total_rewards_white,
                    self.env.steps,
                    agn_b,
                    cgn_b,
                    agn_w,
                    cgn_w,
                ]
            )
        self.current_ep += 1

    def train(self, render_each: int = 1, save_on_learn: bool = True):
        for ep in tqdm.tqdm(range(self.episodes)):
            do_render = ep % render_each == 0
            self.train_episode(do_render)
        self.save()

    def save(self) -> None:
        self.save_learners()

    @abstractmethod
    def save_learners(self):
        pass

    @abstractmethod
    def add_episodes(self, white: Episode, black: Episode) -> None:
        pass
