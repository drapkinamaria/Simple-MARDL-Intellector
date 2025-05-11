from agents.double_agents import DoubleAgentsIntellector
from enviroments.hex_intellector_env import HexIntellectorEnv
from enviroments.second_hex_intellector_env import SecondHexIntellectorEnv
from enviroments.third_hex_intellector_env import ThirdHexIntellectorEnv
from learning import PPO
import os


hidden_layers = (2048, 2048)
num_episodes = 500
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2
learning_rate = 0.003
buffer_size = 2048
batch_size = 128
epochs = 200

envs = [
    ("env1", HexIntellectorEnv),
    ("env2", SecondHexIntellectorEnv),
    ("env3", ThirdHexIntellectorEnv),
]

agents = {}

for env_name, EnvClass in envs:
    result_folder = os.path.join("results", f"DoubleAgents_{env_name}")
    os.makedirs(result_folder, exist_ok=True)

    env = EnvClass()
    env.reset()

    moves_log = os.path.join(result_folder, f"moves_{env_name}.txt")
    ppo = PPO(
        env,
        hidden_layers,
        epochs,
        buffer_size,
        batch_size,
        gamma,
        gae_lambda,
        policy_clip,
        learning_rate,
        log_path=moves_log,
    )

    agent = DoubleAgentsIntellector(
        env=env,
        learner=ppo,
        episodes=num_episodes,
        train_on=buffer_size,
        result_folder=result_folder,
    )
    agent.train(render_each=1, save_on_learn=True)
    agent.save()

    agents[env_name] = agent

    env.close()