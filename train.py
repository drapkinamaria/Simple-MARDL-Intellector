from hex_intellector_env import HexIntellectorEnv
from second_hex_intellector_env import SecondHexIntellectorEnv
from third_hex_intellector_env import ThirdHexIntellectorEnv
from learning import PPO, Episode
import os

save_folder = "./"
os.makedirs(save_folder, exist_ok=True)

hidden_layers = (2048, 2048)
num_episodes = 2000
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2
learning_rate = 0.003
buffer_size = 2048
batch_size = 128
epochs = 500

def make_or_load(name):
    model = PPO(
        env1,
        hidden_layers,
        epochs,
        buffer_size,
        batch_size,
        gamma,
        gae_lambda,
        policy_clip,
        learning_rate,
    )
    path = os.path.join(save_folder, f"{name}.pt")
    if os.path.exists(path):
        model.load(save_folder, name)
    return model


def train_ppo(agents, env, black_model_name, white_model_name, metrics_filename):
    metrics_path = os.path.join(save_folder, metrics_filename)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(
            "episode, winner, "
            "reward_black, reward_white, steps_to_finish, "
            "actor_grad_norm_black, critic_grad_norm_black, "
            "actor_grad_norm_white, critic_grad_norm_white\n"
        )

        black_wins = 0
        white_wins = 0
        draws = 0

        for episode in range(num_episodes):
            step_count = 0
            state = env.reset()
            done = False
            total_reward_black = 0
            total_reward_white = 0
            current_turn = 0  # 0 - black, 1 - white

            for agent in agents:
                agent.logger.log(f"Игра {episode + 1}")

            while not done:
                step_count += 1
                current_agent = agents[current_turn]
                _, _, action_mask = env.get_all_actions(current_turn)
                action, log_prob, value = current_agent.take_action(state, action_mask)
                next_state, rewards, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                reward = rewards[current_turn]
                total_reward_white += rewards[0]
                total_reward_black += rewards[1]

                ep = Episode()
                ep.add(
                    state=state,
                    reward=reward,
                    action=action,
                    goal=done,
                    prob=log_prob,
                    value=value,
                    masks=action_mask,
                )
                current_agent.remember(ep)

                state = next_state
                current_turn = 1 - current_turn

            winner = info.get("winner", None)
            if winner == "black":
                black_wins += 1
            elif winner == "white":
                white_wins += 1
            elif winner == "draw":
                draws += 1

            gb_actor = agents[0].last_actor_grad_norm
            gb_critic = agents[0].last_critic_grad_norm
            gw_actor = agents[1].last_actor_grad_norm
            gw_critic = agents[1].last_critic_grad_norm

            metrics_file.write(
                f"{episode + 1}, {winner}, "
                f"{total_reward_black:.2f}, {total_reward_white:.2f}, "
                f"{step_count}, "
                f"{gb_actor:.6e}, {gb_critic:.6f}, "
                f"{gw_actor:.6e}, {gw_critic:.6f}\n"
            )

            # обучаем после эпизода
            for agent in agents:
                agent.learn()

            agents[0].logger.log_empty_line()

            print(f"[{metrics_filename}] Episode {episode+1}/{num_episodes}")

    # сохраняем модели
    agents[0].save(save_folder, black_model_name)
    agents[1].save(save_folder, white_model_name)


env1 = HexIntellectorEnv()
black_name1 = "ppo_hex_black_first"
white_name1 = "ppo_hex_white_first"
metrics_file1 = "training_metrics_first.txt"
moves_log1 = os.path.join(save_folder, "moves_first.txt")

ppo1_black = PPO(
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

ppo1_white = PPO(
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
train_ppo(
    [ppo1_black, ppo1_white],
    env1,
    black_name1,
    white_name1,
    metrics_file1,
)

env2 = SecondHexIntellectorEnv()
black_name2 = "ppo_hex_black_second"
white_name2 = "ppo_hex_white_second"
metrics_file2 = "training_metrics_second.txt"
moves_log2 = os.path.join(save_folder, "moves_second.txt")

ppo2_black = PPO(
    env2,
    hidden_layers,
    epochs,
    buffer_size,
    batch_size,
    gamma,
    gae_lambda,
    policy_clip,
    learning_rate,
    log_path=moves_log2,
)
ppo2_white = PPO(
    env2,
    hidden_layers,
    epochs,
    buffer_size,
    batch_size,
    gamma,
    gae_lambda,
    policy_clip,
    learning_rate,
    moves_log2,
)

train_ppo(
    [ppo2_black, ppo2_white],
    env2,
    black_name2,
    white_name2,
    metrics_file2,
)

env3 = ThirdHexIntellectorEnv()
ppo3_black = PPO(
    env3,
    hidden_layers,
    epochs,
    buffer_size,
    batch_size,
    gamma,
    gae_lambda,
    policy_clip,
    learning_rate,
    log_path=os.path.join(save_folder, "moves_third.txt"),
)
ppo3_white = PPO(
    env3,
    hidden_layers,
    epochs,
    buffer_size,
    batch_size,
    gamma,
    gae_lambda,
    policy_clip,
    learning_rate,
    log_path=os.path.join(save_folder, "moves_third.txt"),
)
train_ppo(
    [ppo3_black, ppo3_white],
    env3,
    "ppo_hex_black_third",
    "ppo_hex_white_third",
    "training_metrics_third.txt",
)
