from agents.double_agents import DoubleAgentsIntellector
from hex_intellector_env import HexIntellectorEnv
from learning import PPO

hidden_layers = (2048, 2048)
num_episodes = 20
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2
learning_rate = 0.003
buffer_size = 2048
batch_size = 128
epochs = 200

moves_log1 = "moves_first.txt"
env1 = HexIntellectorEnv()
env1.reset()

ppo = PPO(
    env1,
    hidden_layers,
    epochs,
    buffer_size,
    batch_size,
    gamma,
    gae_lambda,
    policy_clip,
    learning_rate,
    log_path=moves_log1,
)

agent = DoubleAgentsIntellector(
        env=env1,
        learner=ppo,
        episodes=100,
        train_on=buffer_size,
        result_folder="results/DoubleAgents",
    )
agent.train(render_each=20, save_on_learn=True)
agent.save()
env1.close()