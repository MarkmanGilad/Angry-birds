import pygame
import sys
from Environment import Environment
from constants import *
import torch
from Human_agent import Human_agent
from ai_agent import DQN_Agent
from State import State
PATH = DEFAULT_MODEL_PATH

def show_game_over(screen):
    font = pygame.font.SysFont("Arial", GAME_OVER_FONT_SIZE)
    text = font.render("GAME OVER", True, RED)
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))

    screen.fill(BLACK)
    screen.blit(text, rect)
    pygame.display.flip()
    pygame.time.wait(GAME_OVER_WAIT_MS)


def main():
    env = Environment()
    state=State()
    env.init_display()
    player = Human_agent()
    # player = DQN_Agent(parametes_path=PATH, env=env) # הוספת ה-env כאן
    run = True

    while run:
        pygame.event.pump()
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                run = False
        if not run:
            break
        
        # 1. קודם כל נותנים לפיזיקה לרוץ (נפילת חזירים/בלוקים בתחילת שלב)
        env.render()
        env.move(None) # מריץ עדכון פיזיקלי בלי ירייה

        # 2. רק אם הציפור לא בתנועה והסביבה התייצבה - הסוכן מקבל החלטה
        # בתוך main.py
        if not env.end_of_game() and env.is_stable():
            state=State()
            state_T=state.toTensor(env)
            # אפשר להעביר את ה-env כאן אם לא הגדרת אותו ב-init
            #action = player.get_action((45, 315),events)
            action = player.get_action(state=state_T, train=False, events=events) 
            env.move(action)
            
        # בדיקת סוף משחק
        if env.end_of_game():
            env.reset()

    pygame.quit()
    sys.exit()
if __name__=='__main__': main()