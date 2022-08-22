import torch
import math

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        #misc agent variables
        self.GAMMA=0.99
        self.LR=1e-4

        #memory
        self.TARGET_NET_UPDATE_FREQ = 1000
        self.REPLAY_BUFFER_SIZE = 100000
        self.BATCH_SIZE = 512
        self.PRIORITY_ALPHA=0.6
        self.PRIORITY_BETA_START=0.4
        self.PRIORITY_BETA_FRAMES = 100000

        #Learning control variables
        self.LEARN_START = 10000
        self.MAX_FRAMES=100000

