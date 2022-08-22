import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env.game_env import Game2048Env
from utils.hyperparameters import Config
from utils.render_episode import RenderEpisode


config = Config()
config.MAX_FRAMES = int(1e5)
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MlpNet(nn.Module):
    def __init__(self, obs_num, action_num):
        super(MlpNet, self).__init__()
        self.obs_num = obs_num
        self.action_num = action_num

        self.fc1 = nn.Linear(self.obs_num, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, self.action_num)
        
    def forward(self, x):
        # x = x.view()  # todo
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):  # todo: 
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, env=None, config=None):
        # hyparameters
        self.device = config.device
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.replay_buffer_size = config.REPLAY_BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START

        # obs and action num
        self.obs_shape = env.observation_space.shape
        self.obs_num = self.obs_shape[0] * self.obs_shape[1]
        self.action_num = env.action_space.n
        self.env = env

        # declare q and target_q networks
        self.q_model = MlpNet(self.obs_num, self.action_num)
        self.q_model = self.q_model.to(self.device)
        self.target_q_model = MlpNet(self.obs_num, self.action_num)
        self.target_q_model = self.target_q_model.to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)

        self.update_count = 0

        # declare memory
        self.memory = ExperienceReplayMemory(self.replay_buffer_size)

        # save highest tile, rewards and losses
        self.episode_highest_tiles = []
        self.episode_rewards = []
        self.losses = []


    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))


    def prep_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        shape = (-1, self.obs_num)  # WIP
        
        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float).view(shape)

        return batch_state, batch_action, batch_reward, batch_next_state


    def computer_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state = batch_vars
        current_q_values = self.q_model(batch_state).gather(1, batch_action)

        # todo: how to deal with done? if done == True, next_state is only state, not None.
        max_next_q_values = self.target_q_model(batch_next_state).max(dim=1)[0].view(-1,1)
        # .max(dim)[0] : return number label; .max(dim)[1] : return value
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        diff = expected_q_values - current_q_values
        loss = self.huber(diff)
        loss = loss.mean()
        return loss


    def select_action(self, s, eps=0.1):  # s is a step
        with torch.no_grad():
            if np.random.random() >= eps:
                s = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.q_model(s).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.action_num)


    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())


    def step_learn(self, s, a, r, s_, frame=0):
        self.append_to_replay(s, a, r, s_)
        
        if frame < self.learn_start:
            return None
        
        batch_vars = self.prep_minibatch()
        loss = self.computer_loss(batch_vars)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        

    def save_loss(self, loss):
        self.losses.append(loss)

    def save_episode_reward(self, episode_reward):
        self.episode_rewards.append(episode_reward)

    def save_model(self):
        torch.save(self.q_model.state_dict(), './rl/save_dqn_agent/q_model.dump')
        torch.save(self.optimizer.state_dict(), './rl/save_dqn_agent/optim.dump')

    def load_model(self):
        pass

    def huber(self, x):
        cond = (x.abs()<1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs()-0.5) * (1-cond)


def main():
    env = Game2048Env()
    agent = DQN(env=env, config=config)

    render_episode = RenderEpisode(env=env)

    episode = 0
    episode_reward = 0
    obs = env.reset()
    obs = obs.reshape(-1)  ## like mujoco

    render_episode.append_episode_state()

    time_start = time.time()

    # training
    for frame_idx in range(1, config.MAX_FRAMES+1):  ## for t in count():
        epsilon = config.epsilon_by_frame(frame_idx)

        action = agent.select_action(obs, epsilon)
        next_obs, reward, done, _ = env.step(action)
        # next_obs = None if done else next_obs  # todo: this is normal
        next_obs = next_obs.reshape(-1)

        render_episode.append_episode_state()

        agent.step_learn(obs, action, reward, next_obs, frame_idx)
        episode_reward += reward

        if done:
            episode += 1

            if episode % 10 == 0:
                render_episode.show_episode()
            else:
                render_episode.reset_episode()

            agent.episode_highest_tiles.append(env.highest())
            print('episode:', episode, 'highest tile:', env.highest())

            obs = env.reset()
            obs = obs.reshape(-1)
            agent.save_episode_reward(episode_reward)
            episode_reward = 0

    # todo: add loss 
    # print('episode reward list:', agent.episode_rewards)
    # print('episode highest tile list:', agent.episode_highest_tiles)
    agent.save_model()
    env.close()

    time_end = time.time()
    time_consume = time_end - time_start
    # print(time_consume)
    print(time.strftime("%H:%M:%S", time.gmtime(time_consume)))


if __name__ == "__main__":
    main()
        
