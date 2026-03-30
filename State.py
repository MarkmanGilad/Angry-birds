import numpy as np
import torch
from constants import *

class State:
    def __init__(self, max_pigs=MAX_PIGS, max_blocks=MAX_BLOCKS):
        self.max_pigs = max_pigs
        self.max_blocks = max_blocks
    
    def build(self, env):
        state_list = []
        
        # 1. נירמול מספר הניסיונות (נניח מקסימום 3)
        state_list.append(env.tries)

        # 2. Pigs state – נירמול מיקומים ביחס לרוחב וגובה המסך
        pig_list = list(env.pigs)
        for pig in pig_list[:self.max_pigs]:
            state_list += [
                pig.rect.centerx / WIDTH,
                pig.rect.centery / HEIGHT
            ]

        # Fill remaining pigs
        for _ in range(self.max_pigs - len(pig_list)):
            state_list += [0.0, 0.0]

        # 3. Blocks state – נירמול מיקומים, גדלים וזוויות
        block_list = list(env.blocks)
        for block in block_list[:self.max_blocks]:
            state_list += [
                block.rect.bottomleft[0] / WIDTH, # x -- bottomleft
                block.rect.bottomleft[1] / HEIGHT,# y - bottomleft
                block.rect.width / BLOCK_SIZE_NORM,   # נניח רוחב מקסימלי סביר
                block.rect.height / BLOCK_SIZE_NORM,  # נניח גובה מקסימלי סביר
                block.angle / BLOCK_ANGLE_NORM,        # נירמול זווית ל-[0,1]
                block.hit / BLOCK_HIT_NORM             # נירמול פגיעות (נהרס ב-2)
            ]
        
        # Fill remaining blocks
        for _ in range(self.max_blocks - len(block_list)):
            state_list += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
        return state_list

    def toTensor(self, env, device=torch.device('cpu')):
        state_list = np.array(self.build(env), dtype=np.float32)
        tensor = torch.from_numpy(state_list).to(device)
        return tensor

    @staticmethod
    def tensor_to_state_list(state_tensor):
        if state_tensor.is_cuda:
            state_tensor = state_tensor.cpu()
        return state_tensor.detach().numpy().tolist()