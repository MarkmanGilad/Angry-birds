import pygame
from constants import *

class Block(pygame.sprite.Sprite):
    def __init__(self, start_pos, width=BLOCK_DEFAULT_WIDTH, height=BLOCK_DEFAULT_HEIGHT, color=BROWN):
        super().__init__()
        self.width = width
        self.height = height
        self.color = color

        self.angle = BLOCK_INITIAL_ANGLE
        self.original_image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.original_image.fill(color)

        self.image = self.original_image
        self.rect = self.image.get_rect(midbottom=start_pos)

        self.mask = pygame.mask.from_surface(self.image)
        self.hit = 0
        self.vy = 0
        self.falling = True  # במצב ברירת מחדל - כולם נופלים

    def update(self):
        pass

    def rotate(self):
        self.angle -= BLOCK_ROTATE_SPEED
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        old_midbottom = self.rect.midbottom
        self.rect = self.image.get_rect(midbottom=old_midbottom)
        self.mask = pygame.mask.from_surface(self.image)

    def fall(self):
        if self.falling:
            x, y = self.rect.midbottom
            self.vy += GRAVITY # כוח כבידה
            y += self.vy
            self.rect.midbottom = (x, y)
        else:
            self.vy = 0 # חשוב לאפס מהירות כשהבלוק נעצר