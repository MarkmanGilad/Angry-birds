import pygame
from constants import *

class Bird(pygame.sprite.Sprite):

    rbird_img=pygame.image.load("img/Red.png")
    rbird_img=pygame.transform.scale(rbird_img,(BIRD_SIZE,BIRD_SIZE))

    def __init__(self, bird="red"):
        super().__init__()
        self.image=Bird.rbird_img
        self.rect=self.image.get_rect()
        self.rect.midbottom=(BIRD_START_X,BIRD_START_Y)
        self.mask=pygame.mask.from_surface(self.image)
        self.vx = 0
        self.vy = 0
        self.move=False
    def Move(self):
        x,y=self.rect.midbottom
        x += self.vx
        y += self.vy
        self.rect.midbottom=x,y
        if self.rect.midbottom!= (BIRD_START_X,BIRD_START_Y):
            self.vy+=GRAVITY
    
    def update(self,x,y):
        self.Move(x,y)
    
    def draw(self,surface):
        surface.blit(self.image,self.rect)