## Установка

```bash
git clone https://github.com/drapkinamaria/Simple-MADRL-Intellector.git
cd Simple-MADRL-Intellector
python -m pip install -r requirements.txt
```

## Запуск обучения

```bash
python train.py
```

## Результаты обучения

Результатом обучения являются два файла: ppo_hex_chess_black.pt и ppo_hex_chess_white.pt. Эти файлы представляют собой модели, разработанные для стратегии игры в Intellector с использованием алгоритма PPO. Одна из этих моделей должна быть загружена в Google Colab: https://colab.research.google.com/drive/1zeifV4r3yirf9RGZL0ufWiRyzbY9xf-w?usp=sharing. Попробуйте сыграть в игру Intellector против искусственного интеллекта.
