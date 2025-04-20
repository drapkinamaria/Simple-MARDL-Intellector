## Установка

```bash
git clone https://github.com/drapkinamaria/Simple-MADRL-Intellector.git
cd Simple-MADRL-Intellector
python -m pip install -r requirements.txt
```

## Входные данные для обучения

Скрытые слои нейронной сети (hidden_layers): Кортеж (1024, 1024), определяющий архитектуру сети (два слоя по 1024 нейрона).

#### Гиперпараметры обучения:
1. epochs=200: Количество эпох обучения на каждом эпизоде.
2. buffer_size=2048: Размер буфера опыта для хранения эпизодов.
3. batch_size=128: Размер батча для обновления модели.
4. gamma=0.99: Коэффициент дисконтирования будущих наград.
5. gae_lambda=0.95: Параметр для обобщенного оценивания преимущества (GAE).
6. policy_clip=0.2: Ограничение изменения политики.
7. learning_rate=0.003: Скорость обучения.

## Запуск обучения

```bash
python train.py
```

## Результаты обучения

Результатом обучения являются два файла: ppo_hex_chess_black.pt и ppo_hex_chess_white.pt. Эти файлы представляют собой модели, разработанные для стратегии игры в Intellector с использованием алгоритма PPO. Одна из этих моделей должна быть загружена в Google Colab: https://colab.research.google.com/drive/1zeifV4r3yirf9RGZL0ufWiRyzbY9xf-w?usp=sharing. Попробуйте сыграть в игру Intellector против искусственного интеллекта.
