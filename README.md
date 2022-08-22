# 2048 RL

## 2048

[2048](https://play2048.co/) is a single-player sliding tile puzzle game. The objective of the game is to slide numbered tiles on a grid to combine them to create a tile with the number 2048(math: 2048 = 2**11).

`env` uses an open-source 2048 game in OpenAI gym interface. Thanks for authors @dsgiitr, @rgal and @yangrui.

## RL Algorithm

`random_agent.py` - Choose a random action to play 2048.

`dqn.py` - Train a dqn agent to play 2048.
