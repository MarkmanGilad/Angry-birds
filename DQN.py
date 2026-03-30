import math
import random
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from Environment import Environment
from constants import *
from State import *
import os
HuberLoss = nn.SmoothL1Loss()
env=Environment()
input_size = 1 + 2*MAX_PIGS + 6*MAX_BLOCKS
layer1 = DQN_LAYER1
layer2 = DQN_LAYER2
output_size = NUM_ACTIONS
gamma = GAMMA 
MSELoss = nn.MSELoss()
class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if torch.cuda.is_available:
            self.device = torch.device('cpu') # 'cuda'
        else:
            self.device = torch.device('cpu')
        
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.output = nn.Linear(layer2, output_size)
        
    def forward (self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    import os   # אם אין למעלה בקובץ

    def save_params(self, path):
        dir_name = os.path.dirname(path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.state_dict(), path)

    def copy (self):
        new_DQN = DQN()
        new_DQN.load_state_dict(self.state_dict())
        return new_DQN
    
    def loss(self, Q_value, rewards, Q_next_Values, Dones):
        # חישוב ה-Target לפי משוואת בלמן
        Q_new = rewards + gamma * Q_next_Values * (1 - Dones)
        
        # שימוש ב-Huber Loss במקום MSE
        return HuberLoss(Q_value, Q_new)
