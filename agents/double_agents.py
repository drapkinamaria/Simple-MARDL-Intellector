from copy import deepcopy
from agents.base import BaseAgent
from buffer.episode import Episode
from learnings.base import Learning
from hex_intellector_env import HexIntellectorEnv

class DoubleAgentsIntellector(BaseAgent):
    def __init__(
        self,
        env: HexIntellectorEnv,
        learner: Learning,
        episodes: int,
        train_on: int,
        result_folder: str,
    ) -> None:
        super().__init__(env, learner, episodes, train_on, result_folder)
        self.white_agent = deepcopy(learner)
        self.black_agent = deepcopy(learner)

    def add_episodes(self, white: Episode, black: Episode) -> None:
        self.white_agent.remember(white)
        self.black_agent.remember(black)

    def learn(self):
        self.white_agent.learn()
        self.black_agent.learn()

    def save_learners(self):
        self.white_agent.save(self.result_folder, "white_ppo")
        self.black_agent.save(self.result_folder, "black_ppo")