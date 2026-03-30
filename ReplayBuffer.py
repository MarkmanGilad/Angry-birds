from collections import deque
import random
import torch
import numpy as np
from State import State


from constants import *


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state_T, action, reward, next_state_T, done):
        # יצירת טנזורים לשאר הערכים
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).reshape(1)
        done_tensor = torch.tensor(done, dtype=torch.float32).reshape(1)
        self.buffer.append((state_T, action_tensor, reward_tensor, next_state_T, done_tensor))
    
    def push_tensors (self, state_tensor, action_tensor, reward_tensor, next_state_tensor, done):
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done))
            
    def sample (self, batch_size):
        if (batch_size > self.__len__()):
            batch_size = self.__len__()
        state_tensors, action_tensor, reward_tensors, \
                   next_state_tensors, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.vstack(state_tensors)
        actions = torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_states = torch.vstack(next_state_tensors)
        done_tensor = torch.tensor(dones).long().reshape(-1,1)
        return states, actions, rewards, next_states, done_tensor

    def __len__(self):
        return len(self.buffer)