import pygame
from constants import *

class Human_agent:
    def __init__(self):
        pass
    def get_action (self,pos=(BIRD_START_X, BIRD_START_Y),events=None,state=None, train = False):
        
        for event in events:
            # if event.type==pygame.MOUSEBUTTONDOWN:
            #     self.p1=pygame.mouse.get_pos()
            
            if event.type==pygame.MOUSEBUTTONUP:
                pos2= pygame.mouse.get_pos()
                dx = pos[0] - pos2[0]  # drag left = positive
                dy = pos2[1] - pos[1]  # drag down = positive
                vx = int(dx * (ACTION_COMPONENTS - 1) / BIRD_START_X)
                vy = int(dy * (ACTION_COMPONENTS - 1) / (HEIGHT - BIRD_START_Y))
                vx = max(0, min(vx, ACTION_COMPONENTS - 1))
                vy = max(0, min(vy, ACTION_COMPONENTS - 1))
                return vx,vy
        return None