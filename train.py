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

num = 11

epochs = TRAIN_EPOCHS
C = TARGET_UPDATE_INTERVAL
batch = TRAIN_BATCH_SIZE
path = DEFAULT_MODEL_PATH

def train():
    wandb.init(
        project="angry-birds-dqn",
        name=f"angry-birds_{num}",
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
    env.init_display(title=f"Angry Birds - Test {num}")
    player = DQN_Agent(env=env)
    replay = ReplayBuffer()
    Q = player.DQN
    Q_hat: DQN = Q.copy()
    Q_hat.train = False
    optim = torch.optim.Adam(Q.parameters(), lr=TRAIN_LR)
    
    success_rate = []
    current_epoch_losses = []
    good = False
    for epoch in range(epochs):
        if epoch % C == 0: 
            success_rate.append(0)
        
        env.reset()
        pigs = len(env.pigs)
        initial_pigs = pigs
        tries = env.tries
        episode_score = 0
        episode_reward_sum = 0
        shots_fired = 0
        pigs_killed = 0
        episode_losses = []
        state_T=state.toTensor(env)
        done = False
        while not done: # nor done:
            action = player.get_action(state_T, epoch=epoch, train=True)
            reward, done = env.move(action)
            shots_fired += 1
            episode_score += SCORE_BIRD_FIRED
            
            while not done and (env.bird.move or not env.is_stable()):
                reward, done = env.move(None)
                env.render()
            next_state_T = state.toTensor(env)
            
            new_pigs = len(env.pigs)
            killed_this_shot = pigs - new_pigs
            pigs_killed += killed_this_shot
            episode_score += killed_this_shot * SCORE_PIG_KILLED
            pigs = new_pigs
            tries = env.tries
            episode_reward_sum += reward
            
            replay.push(state_T, action, reward, next_state_T, done)
            state_T = next_state_T.clone()
            
            if epoch < TARGET_UPDATE_INTERVAL:
                continue
                
            states, actions, rewards, next_states, dones = replay.sample(batch)
            
            # --- DDQN ---
            # Q(s, a) for the actions actually taken
            all_Q = Q(states)                          # [batch, 100]
            action_indices = (actions[:, 0] * ACTION_COMPONENTS + actions[:, 1]).long()  # [batch]
            Q_values = all_Q.gather(1, action_indices.unsqueeze(1))  # [batch, 1]
            
            with torch.no_grad():
                # Online net selects best action for next state
                next_Q_online = Q(next_states)                      # [batch, 100]
                next_best_indices = next_Q_online.argmax(dim=1, keepdim=True)  # [batch, 1]
                # Target net evaluates that action
                next_Q_target = Q_hat(next_states)                  # [batch, 100]
                Q_hat_Values = next_Q_target.gather(1, next_best_indices)  # [batch, 1]
            
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            
            # --- הוספה: שמירת ערך ה-loss הנוכחי ---
            current_epoch_losses.append(loss.item())
            episode_losses.append(loss.item())
            # ------------------------------------
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=GRAD_MAX_NORM)
            optim.step()
            optim.zero_grad()

        if pigs == 0: 
            success_rate[int(epoch/C)] += 1
            episode_score += SCORE_WIN
        elif tries == 0:
            episode_score += SCORE_LOSS
        
        epsilon = player.epsilon_greedy(epoch)
        episode_log = {
            "episode/score": episode_score,
            "episode/reward_sum": episode_reward_sum,
        }
        avg_ep_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
        if episode_losses:
            episode_log["episode/loss"] = avg_ep_loss
        wandb.log(episode_log, step=epoch)
        
        win_str = "WIN" if pigs == 0 else "LOSS"
        print(f"[Test {num}] Epoch {epoch} | {win_str} | Score: {episode_score} | Reward: {episode_reward_sum:.2f} | Loss: {avg_ep_loss:.4f} | Pigs killed: {pigs_killed}/{initial_pigs} | Tries left: {tries} | Shots: {shots_fired} | Eps: {epsilon:.4f}")
            
        if epoch % C == 0 and epoch != 0:
            Q_hat.load_state_dict(Q.state_dict())
            current_epoch_losses = []
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

    # plt.tight_layout()
    # plt.show()
if __name__ == "__main__":
    train()