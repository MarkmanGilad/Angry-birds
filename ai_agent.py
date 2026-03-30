import math
import random
import torch
import torch.nn as nn
import numpy as np
from DQN import DQN
from constants import *
from State import State


epsilon_start = EPSILON_START
epsilon_final = EPSILON_FINAL
epsiln_decay = EPSILON_DECAY#ככל שמגדילים, יש יותר זמן אקראי

# epochs = 1000
# batch_size = 64
gamma = GAMMA 
MSELoss = nn.MSELoss()
class DQN_Agent:
    def __init__(self, parametes_path = None, train = True, env= None) -> None:
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train(train)
        self.env = env

    def train (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_action (self, state_T, epoch = 0, events= None, train = True):
        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()
        actionsx = torch.arange(ACTION_COMPONENTS)
        actionsy = torch.arange(ACTION_COMPONENTS)
        if train and rnd < epsilon:
            return random.choice(actionsx),random.choice(actionsy)
        # xx, yy = torch.meshgrid(actionsx, actionsy, indexing="ij")
        # action_pairs = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        # expand_state = state_tensor.unsqueeze(0).repeat(action_pairs.shape[0], 1)

        with torch.no_grad():
            Q_values = self.DQN(state_T)
        # shape: [100]
        best_idx = torch.argmax(Q_values)
        best_action = self.index_to_action(best_idx)
        return best_action
    
    def index_to_action(self, index):
        x = index // ACTION_COMPONENTS
        y = index % ACTION_COMPONENTS
        return x, y

    def action_to_index(self, action):
        return action[0] * ACTION_COMPONENTS + action[1]

    def get_actions (self, states, dones, train = True):
        actions = []
        for i, state in enumerate(states):
            actions.append(self.get_action(state, train=train)) #SARSA = True / Q-learning = False
        return torch.tensor(actions)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None, train=True, env=None):
        return self.get_action(state=state, train=train)
