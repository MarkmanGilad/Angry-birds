import pygame
from constants import *
class graphics:
    def __init__(self):
        pygame.init()
        self.screen=pygame.display.set_mode((WIDTH,HEIGHT))
        self.image=pygame.image.load("img/background.webp")
        self.rug=pygame.image.load("img/rug.png")
        self.image=pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rug=pygame.transform.scale(self.rug,(RUG_SIZE,RUG_SIZE))
        self.main_surf = pygame.Surface((WIDTH, HEIGHT))
        self.agent=None
    def render(self):
        self.main_surf.blit(self.image,(0,0))
        self.main_surf.blit(self.rug,(RUG_X,RUG_Y))
        self.screen.blit(self.main_surf,(0,0))
        pygame.display.update()
    def draw_agent(self, agent):
        # Draw the agent's image on the screen at the agent's position
        self.screen.blit(agent.image, agent.rect)