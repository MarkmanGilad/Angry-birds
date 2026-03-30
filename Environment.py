import pygame
from Bird import Bird
from constants import *
from Block import Block
from Pig import Pig
import random
from State import State 
import math

class Environment:
    def __init__(self):
        self.bird = Bird()
        self.pigs = pygame.sprite.Group()
        self.blocks = pygame.sprite.Group()
        self.tries = INITIAL_TRIES
        self.level = INITIAL_LEVEL
        self.screen = None
        self.reward=0
        self.steps_since_shot = 0

    def init_pigs (self,pos):
        pig=Pig(pos)
        self.pigs.add(pig)

    def init_blocks (self):
        block=Block((500,310))
        self.blocks.add(block)
        block=Block((300,310))
        self.blocks.add(block)

    def init_level(self, level_num):
        self.tries = INITIAL_TRIES
        self.pigs.empty()
        self.blocks.empty()

        num_buildings = LEVEL_NUM_BUILDINGS

        for _ in range(num_buildings):
            x = random.randint(LEVEL_BUILDING_X_MIN, LEVEL_BUILDING_X_MAX)
            num_floors = 1
            
            # משתנה שיעזור לנו לדעת מה הגובה המצטבר של המבנה
            current_top_y = GROUND_Y 

            for i in range(num_floors):
                is_horizontal = random.random() < LEVEL_HORIZONTAL_CHANCE
                width = LEVEL_HORIZONTAL_WIDTH if is_horizontal else BLOCK_DEFAULT_WIDTH
                height = LEVEL_HORIZONTAL_HEIGHT if is_horizontal else BLOCK_DEFAULT_HEIGHT
                
                # מיקום הבלוק: התחתית שלו היא ה-top של הקומה הקודמת
                block = Block((x, current_top_y), width=width, height=height)
                self.blocks.add(block)
                
                # עדכון הגובה לקומה הבאה
                current_top_y -= height

            # מיקום החזיר: בדיוק על הגג של הקומה האחרונה
            # current_top_y כרגע מייצג את ה-top של הבלוק העליון ביותר
            self.init_pigs((x, current_top_y))

    def init_display(self, title="Angry Birds"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("img/background.webp")
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        self.rug = pygame.image.load("img/rug.png")
        self.rug = pygame.transform.scale(self.rug, (RUG_SIZE, RUG_SIZE))
        self.init_level(self.level)

    def calculate_ballistic_distance(self,x0, y0,action, x, y):
        vx=(action[0] + 1) * VX_SCALE
        vy=(action[1] - 1) * VY_SCALE
        yp=y0+vy*(x-x0)/vx + (GRAVITY/2)*((x-x0)**2)/vx**2
        if abs(y-yp)<1: return 1
        return abs(y-yp)
        
    def get_state(self):
        state=State()
        state.toTensor(self)
        return state
    def force_stabilize_blocks(self):
        for block in self.blocks:
            # מציאת הזווית הקרובה ביותר בקפיצות של 90 מעלות
            current_angle = block.angle % 360
            snapped_angle = round(current_angle / 90) * 90
            block.angle = snapped_angle
            
            # איפוס מהירויות פיזיקליות
            block.vy = 0
            block.falling = False

    def move(self, action):
        # — אם יש פעולה (action לא None) — בצע ירייה / התחל תנועה
        
        done = False
        pigs_num_before_step = len(self.pigs) # כמות החזירים בתחילת הצעד הנוכחי
        
        if action is not None:
            if not self.bird.move:
                # איפוס reward רק כשמתחילים ירייה חדשה
                self.reward = 0
                # שמירת כמות החזירים ברגע הירייה כדי לבדוק פגיעה בהמשך
                self.pigs_before_shot = len(self.pigs)
                
                self.bird.vx = (action[0] + 1) * VX_SCALE
                self.bird.vy = (action[1] - 1) * VY_SCALE
                self.bird.move = True
                
                self.reward -= REWARD_SHOOT 
                self.tries -= 1

        # תנועת הציפור
        if self.bird.move:
            check = True
            for block in list(self.blocks):
                if 270 < block.angle < BLOCK_INITIAL_ANGLE:
                    check = False
            if check:
                self.bird.Move()

        # גיבוש קבוצות sprites לציפור / חזירים
        bird_group = pygame.sprite.GroupSingle(self.bird)

        # התנגשויות ציפור-חזירים
        killed = pygame.sprite.groupcollide(bird_group, self.pigs, False, True, pygame.sprite.collide_mask)
        
        # עדכון חזירים: נפילה, בדיקות קרקע וכו׳
        for pig in list(self.pigs):
            pig.stay = False
            for block in self.blocks:
                if pygame.sprite.collide_mask(pig, block):
                    pig.stay = True
                    break
            if not pig.stay:
                pig.Fall()
            if pig.rect.bottom >= GROUND_Y:
                pig.stay = True
                pig.kill()

        # --- עדכון בלוקים: לוגיקת נפילה משופרת ---
        
        # שלב א': נמיין את הבלוקים מלמטה למעלה (לפי Y) כדי שנוכל לבדוק יציבות מהקרקע מעלה
        sorted_blocks = sorted(list(self.blocks), key=lambda b: b.rect.bottom, reverse=True)
        
        for block in sorted_blocks:
            # נניח בתחילה שהבלוק נופל
            block.falling = True
            
            # אם הבלוק על הרצפה - הוא יציב
            if block.rect.bottom >= GROUND_Y:
                block.rect.bottom = GROUND_Y # הצמדה לרצפה
                block.falling = False
            else:
                # אם הוא לא על הרצפה, נבדוק אם הוא יושב על בלוק אחר שכבר קבענו שהוא לא נופל
                for other in sorted_blocks:
                    if block is other or other.falling:
                        continue # אי אפשר להישען על בלוק שבעצמו נופל
                    
                    # בדיקה אם התחתית של הבלוק הנוכחי נוגעת בחלק העליון של הבלוק השני
                    if pygame.sprite.collide_mask(block, other):
                        # בדיקה שהבלוק מעל השני (עם טווח טעות קטן)
                        if abs(block.rect.bottom - other.rect.top) < BLOCK_SNAP_TOLERANCE:
                            block.falling = False
                            # הצמדה מדויקת כדי למנוע "רעידות"
                            block.rect.bottom = other.rect.top
                            break

            # הרצת הנפילה/סיבוב הפיזיקלי
            if block.falling:
                block.fall()
            else:
                block.vy = 0 # איפוס מהירות אם הוא יציב

            # טיפול בהתנגשות עם הציפור (נשאר דומה)
            if pygame.sprite.collide_mask(block, self.bird):
                for pig in list(self.pigs):
                    if pygame.sprite.collide_mask(block, pig):
                        self.reward += REWARD_BLOCK_PIG_COLLIDE
                block.rect.midbottom = (block.rect.midbottom[0] + self.bird.vx * 2 + BIRD_SIZE,
                                        block.rect.midbottom[1])
                # סימון הבלוק שיתחיל ליפול/להסתובב אחרי המכה
                block.angle -= 1 
                block.hit += 1
                self.bird.rect.midbottom = (BIRD_START_X, BIRD_START_Y)
                self.bird.move = False

            # סיבוב בלוקים פגועים
            if 270 < block.angle < BLOCK_INITIAL_ANGLE:
                block.rotate()
            
            if block.hit >= BLOCK_MAX_HITS:
                block.kill()    

        if not self.is_stable():
            self.steps_since_shot += 1
        # בדיקה אם עבר יותר מדי זמן ללא התייצבות
        if self.steps_since_shot > MAX_STEPS_SINCE_SHOT:
            self.force_stabilize_blocks()
            self.bird.move = False
            self.bird.rect.midbottom = (BIRD_START_X, BIRD_START_Y)
            self.steps_since_shot = 0        
        # ציפור נופלת לקרקע (פספוס מוחלט)
        if self.bird.rect.midbottom[1] > BIRD_OUT_BOTTOM or self.bird.rect.midbottom[0] > BIRD_OUT_RIGHT:
            if hasattr(self, 'pigs_before_shot'):
                if len(self.pigs) == self.pigs_before_shot:
                    self.reward += REWARD_MISS_PENALTY
                del self.pigs_before_shot
            
            self.bird.rect.midbottom = (BIRD_START_X, BIRD_START_Y)
            self.bird.move = False

        next_state = self.get_state()
        # חישוב בונוס על חזירים שנהרגו בפריים הזה
        pigs_killed_this_frame = pigs_num_before_step - len(self.pigs)
        self.reward += pigs_killed_this_frame * REWARD_PIG_KILL
        if pigs_killed_this_frame > 0 and len(self.pigs) == 0:
            self.reward += REWARD_ALL_PIGS_DEAD
        if self.end_of_game(): 
            done = True
        if self.tries == 0 and len(self.pigs) > 0: 
            self.reward = REWARD_LOSS_PENALTY
        return self.reward, done
    
    def is_stable(self):
        # בדיקה אם הציפור בתנועה
        if self.bird.move:
            return False
        # בדיקה אם יש חזיר שנופל
        for pig in self.pigs:
            if not pig.stay:
                return False
        # בדיקה אם יש בלוק שנופל או מסתובב
        for block in self.blocks:
            if block.falling or (block.angle < BLOCK_INITIAL_ANGLE and block.angle > 270):
                return False
        return True
    
    def render (self):
        # draw background to clear
        # draw rugs on screen
        # draw blocks on screen
        # draw pigs on screen
        # draw bird on screen
                
        self.screen.blit(self.background,(0,0))
        self.screen.blit(self.rug,(RUG_X,RUG_Y))
        self.bird.draw(self.screen)
        self.blocks.draw(self.screen)
        self.pigs.draw(self.screen)
        self.draw_tries()
        
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(FPS)

    def draw_tries(self):
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(f"Tries: {self.tries}", True, BLACK)
        self.screen.blit(text, (10, 10))

    def end_of_game (self):
        if self.bird.move:
            return False
        for block in self.blocks:
            if block.angle<BLOCK_INITIAL_ANGLE and block.angle>270:
                return False
        if len(self.pigs)==0:
            return True
        if self.tries==0: 
            return True
        return False
    
    def reset(self):
        # אתחל את הסביבה מחדש
        self.level = INITIAL_LEVEL
        self.tries = INITIAL_TRIES
        self.pigs.empty()
        self.blocks.empty()
        # אתחל HUD / ריסט של bird
        self.bird = Bird()
        # אתחול מחדש של stage/level
        self.init_level(self.level)
        # אם צריך — גם אתחול של display / screen
        # self.init_display()  # תלוי אם אתה רוצה לפתוח חלון מחדש
        # החזר וקטור מצב התחלתי  
        return self.get_state()
    def is_win(self):
        if len(self.pigs)==0: return True
        return False