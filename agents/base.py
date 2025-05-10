from abc import ABC, abstractmethod
import os
import csv
import numpy as np
import tqdm

from learnings.base import Learning
from buffer.episode import Episode
from hex_intellector_env import HexIntellectorEnv
import intellector.pieces as Pieces

class BaseAgent(ABC):
    def __init__(
        self,
        env: HexIntellectorEnv,
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
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(os.path.join(self.result_folder, "renders"), exist_ok=True)

        self.current_ep = 0
        self.grad_norms = {
            "white_actor": [], "white_critic": [],
            "black_actor": [], "black_critic": [],
        }

        # CSV-лог
        self.log_path = os.path.join(self.result_folder, "episode_logs.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "winner",
                    "reward_black", "reward_white", "steps_to_finish",
                    "actor_grad_norm_black", "critic_grad_norm_black",
                    "actor_grad_norm_white", "critic_grad_norm_white"
                ])

    def take_action(self, turn: int, episode: Episode):
        mask = self.env.get_all_actions(turn)[-1]
        state = self.env.get_state(turn)
        action, log_prob, value = self.learner.take_action(state, mask)
        next_state, rewards, done, truncated, info = self.env.step(action)

        self.learner.logger.log(
            f"Ep {self.current_ep + 1} | Turn {turn} | "
            f"Action {action} | Reward {rewards[turn]:.4f} | Done {done}"
        )

        episode.add(state, rewards[turn], action, done, log_prob, value, mask)
        return done, next_state, rewards, action, log_prob, value, mask, info

    def train_episode(self, render: bool):
        # Сбрасываем рендеры и истории
        renders = []

        def _render():
            if self.env.render_mode != "human":
                renders.append(self.env.render())

        # Создаём эпизоды для каждого игрока
        ep_w = Episode()
        ep_b = Episode()

        # Сбрасываем среду
        self.env.reset()
        turn = Pieces.WHITE

        # Рендерим стартовую позицию
        _render()

        # Очищаем накопленные нормы перед эпизодом
        self.grad_norms = {k: [] for k in self.grad_norms}

        # Проигрываем эпизод
        while True:
            done, *data = self.take_action(
                turn,
                ep_w if turn == Pieces.WHITE else ep_b
            )
            _render()
            if done:
                # из последнего take_action возьмём info
                info = data[-1]
                break
            turn = 1 - turn

        # Учимся и сохраняем градиенты внутрь self.grad_norms
        self.add_episodes(ep_w, ep_b)
        self.learn()

        # Сохраняем модели, если пора
        if (self.current_ep + 1) % self.train_on == 0:
            self.save()

        # --- Логируем результаты эпизода ---
        # Номер эпизода
        episode_num = self.current_ep + 1

        # Победитель из info["winner"]
        widx = info.get("winner")
        if widx == Pieces.WHITE:
            winner = "white"
        elif widx == Pieces.BLACK:
            winner = "black"
        else:
            winner = "draw"

        # Награды
        reward_white = sum(ep_w.rewards)
        reward_black = sum(ep_b.rewards)
        # Шаги до конца
        steps_to_finish = self.env.steps

        # Средние нормы градиентов
        agn_b = np.mean(self.grad_norms["black_actor"]) if self.grad_norms["black_actor"] else 0.0
        cgn_b = np.mean(self.grad_norms["black_critic"]) if self.grad_norms["black_critic"] else 0.0
        agn_w = np.mean(self.grad_norms["white_actor"]) if self.grad_norms["white_actor"] else 0.0
        cgn_w = np.mean(self.grad_norms["white_critic"]) if self.grad_norms["white_critic"] else 0.0

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_num, winner,
                reward_black, reward_white, steps_to_finish,
                agn_b, cgn_b,
                agn_w, cgn_w
            ])

        # Инкремент счётчика эпизодов
        self.current_ep += 1

    def train(self, render_each: int=1, save_on_learn: bool = True):
        for ep in tqdm.tqdm(range(self.episodes)):
            do_render = (ep % render_each == 0)
            self.train_episode(do_render)
            self.save()

    def save(self):
        self.save_learners()

    @abstractmethod
    def save_learners(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def add_episodes(self, white: Episode, black: Episode) -> None:
        pass
