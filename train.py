from agents.double_agents import DoubleAgentsIntellector
from enviroments.hex_intellector_env import HexIntellectorEnv
from enviroments.second_hex_intellector_env import SecondHexIntellectorEnv
from enviroments.third_hex_intellector_env import ThirdHexIntellectorEnv
from learning import PPO
import os


hidden_layers = (512, 512)
num_episodes = 100000
learning_rate = 3e-4
policy_clip = 0.1
gamma = 0.99
gae_lambda = 0.95
buffer_size = 8192
batch_size = 512
epochs = 4
train_on = 5


envs = [
    # ("env1", HexIntellectorEnv),
    # ("env2", SecondHexIntellectorEnv),
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
        train_on=train_on,
        result_folder=result_folder,
    )
    agent.train(render_each=1, save_on_learn=True)
    agent.save()

    agents[env_name] = agent

    env.close()
