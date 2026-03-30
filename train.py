from DQN import DQN
from ai_agent import DQN_Agent
from Environment import Environment
from ReplayBuffer import ReplayBuffer
from State import State
from constants import *
# import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch 

num = 1

epochs = TRAIN_EPOCHS
C = TARGET_UPDATE_INTERVAL
batch = TRAIN_BATCH_SIZE
path = DEFAULT_MODEL_PATH

def train():
    wandb.init(
        project="angry-birds-dqn",
        name=f"test_{num}",
        config={
            "epochs": TRAIN_EPOCHS,
            "target_update_interval": TARGET_UPDATE_INTERVAL,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": TRAIN_LR,
            "grad_max_norm": GRAD_MAX_NORM,
            "win_threshold": WIN_THRESHOLD,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_final": EPSILON_FINAL,
            "epsilon_decay": EPSILON_DECAY,
            "num": num,
        },
    )

    state = State()
    env = Environment()
    env.init_display()
    player = DQN_Agent(env=env)
    replay = ReplayBuffer()
    Q = player.DQN
    Q_hat: DQN = Q.copy()
    Q_hat.train = False
    optim = torch.optim.Adam(Q.parameters(), lr=TRAIN_LR)
    
    success_rate = []
    # --- הוספה: רשימות למעקב אחר loss ---
    loss_history = []
    current_epoch_losses = [] 
    # -----------------------------------
    good = False
    for epoch in range(epochs):
        if epoch % C == 0: 
            success_rate.append(0)
        
        env.reset()
        pigs = len(env.pigs)
        tries = env.tries
        print(epoch, end="\r")
        state_T=state.toTensor(env)
        done = False
        while not done: # nor done:
            action = player.get_action(state_T, epoch=epoch, train=True)
            env.move(action)
            
            while env.bird.move or not env.is_stable():
                reward, done = env.move(None)
                env.render()
            next_state_T = state.toTensor(env)
            
            pigs = len(env.pigs)
            tries = env.tries
            
            replay.push(state_T, action, reward, next_state_T, done)
            state_T = next_state_T.clone()
            
            if epoch < TARGET_UPDATE_INTERVAL:
                continue
                
            states, actions, rewards, next_states, dones = replay.sample(batch)
            Q_values = Q(states, actions)
            next_actions = player.get_actions(next_states, dones, train=True)
            
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions)
            
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            
            # --- הוספה: שמירת ערך ה-loss הנוכחי ---
            current_epoch_losses.append(loss.item())
            # ------------------------------------
            
            wandb.log({"step_loss": loss.item()})
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=GRAD_MAX_NORM)
            optim.step()
            optim.zero_grad()

        if pigs == 0: 
            success_rate[int(epoch/C)] += 1
            
        if epoch % C == 0 and epoch != 0:
            Q_hat.load_state_dict(Q.state_dict())
            
            # --- הוספה: חישוב ממוצע ה-loss לבלוק של 500/1000 איטרציות ---
            if current_epoch_losses:
                avg_loss = sum(current_epoch_losses) / len(current_epoch_losses)
                loss_history.append(avg_loss)
                current_epoch_losses = [] # איפוס לרשימה הבאה
            else:
                loss_history.append(0)
            
            wins = success_rate[int(epoch/C-1)]
            avg_loss = loss_history[-1]
            print("epoch:", epoch, "wins:", wins, "avg loss:", avg_loss)
            wandb.log({
                "epoch": epoch,
                "interval_wins": wins,
                "interval_win_rate": wins / C,
                "interval_avg_loss": avg_loss,
            })
            if wins > WIN_THRESHOLD*C:
                player.save_param(path)
                good = True
    if not good:    
        player.save_param(path)
    wandb.finish()
    # # עדכון הקריאה לפונקציית הציור
    # plot_results(success_rate, loss_history)

# # --- עדכון פונקציית הגרפים ---
# def plot_results(success_rate, loss_history):
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#     
#     # גרף ניצחונות
#     ax1.plot(success_rate, color='green', label='Wins')
#     ax1.set_title('Wins per Interval')
#     ax1.set_xlabel(f'Intervals (per {C} epochs)')
#     ax1.set_ylabel('Number of Wins')
#     ax1.grid(True)
#     ax1.legend()
# 
#     # גרף Loss
#     ax2.plot(loss_history, color='red', label='Loss')
#     ax2.set_title('Average Loss per Interval')
#     ax2.set_xlabel(f'Intervals (per {C} epochs)')
#     ax2.set_ylabel('Loss Value')
#     ax2.grid(True)
#     ax2.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    train()