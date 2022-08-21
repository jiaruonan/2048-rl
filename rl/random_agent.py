import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from env.game_env import Game2048Env


class RandomAgent:
    def __init__(self, env=None):
        self.action_num = env.action_space.n
        
        # save highest tile, rewards and losses
        self.episode_highest_tiles = []
        self.episode_rewards = []
        
    def select_action(self):
        return np.random.randint(0, self.action_num)


def main():
    env = Game2048Env()
    agent = RandomAgent(env=env)

    episode = 0
    episode_reward = 0
    env.reset()

    time_start = time.time()

    # valuation
    for frame_idx in range(1, 100000):

        action = agent.select_action()
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            episode += 1
            agent.episode_highest_tiles.append(env.highest())
            print('episode:', episode, 'highest tile:', env.highest())
            env.reset()
            agent.episode_rewards.append(episode_reward)
            episode_reward = 0

    # print('episode reward list:', agent.episode_rewards)
    # print('episode highest tile list:', agent.episode_highest_tiles)
    env.close()

    time_end = time.time()
    time_consume = time_end - time_start
    # print(time_consume)
    print(time.strftime("%H:%M:%S", time.gmtime(time_consume)))


if __name__ == "__main__":
    main()

