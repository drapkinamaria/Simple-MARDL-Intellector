from hex_chess_env import HexChessEnv
from learning import PPO, Episode
import numpy as np
import os

# Пути для сохранения моделей
save_folder = "./"
black_model_name = "ppo_hex_chess_black"
white_model_name = "ppo_hex_chess_white"

# Инициализация среды и агентов
env = HexChessEnv()
hidden_layers = (1024, 1024)

# Если существуют сохранённые модели, загружаем их, иначе создаём новых агентов
if os.path.exists(
    os.path.join(save_folder, f"{black_model_name}.pt")
) and os.path.exists(os.path.join(save_folder, f"{white_model_name}.pt")):
    ppo_agent_black = PPO(
        env,
        hidden_layers,
        epochs=200,
        buffer_size=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        learning_rate=0.003,
    )
    ppo_agent_white = PPO(
        env,
        hidden_layers,
        epochs=200,
        buffer_size=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        learning_rate=0.003,
    )
    ppo_agent_black.load(save_folder, black_model_name)
    ppo_agent_white.load(save_folder, white_model_name)
else:
    ppo_agent_black = PPO(
        env,
        hidden_layers,
        epochs=200,
        buffer_size=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        learning_rate=0.003,
    )
    ppo_agent_white = PPO(
        env,
        hidden_layers,
        epochs=200,
        buffer_size=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        learning_rate=0.003,
    )


def train_ppo(agents, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward_black = 0
        total_reward_white = 0
        current_turn = 0  # 0 - черные, 1 - белые

        while not done:
            current_agent = agents[current_turn]
            action_mask = np.ones(env.action_space.n)
            action, log_prob, value = current_agent.take_action(state, action_mask)
            next_state, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = rewards[current_turn]
            total_reward_black += rewards[0]
            total_reward_white += rewards[1]

            current_episode = Episode()
            current_episode.add(
                state=state,
                reward=reward,
                action=action,
                goal=done,
                prob=log_prob,
                value=value,
                masks=action_mask,
            )
            current_agent.remember(current_episode)

            state = next_state
            current_turn = 1 - current_turn  # Смена хода

        # Обучение каждого агента после завершения эпизода
        for agent in agents:
            agent.learn()

        print(
            f"Episode {episode + 1}/{num_episodes}, Reward Black: {total_reward_black}, Reward White: {total_reward_white}"
        )

    # Сохранение моделей агентов
    agents[0].save(save_folder, black_model_name)
    agents[1].save(save_folder, white_model_name)


# Агенты в массиве: первый агент - черные, второй - белые
agents = [ppo_agent_black, ppo_agent_white]

# Запуск обучения
train_ppo(agents, env, num_episodes=2)
