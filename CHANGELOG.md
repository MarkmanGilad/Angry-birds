# Changelog

## 2026-03-30 (session 2)

### Added wandb logging to `train.py`
- Added `import wandb` and `num = 1` variable at the top to name each training run (`test_{num}`).
- `wandb.init()` logs all hyperparameters from `constants.py` (epochs, lr, gamma, epsilon, etc.).
- Per-episode logging: `episode/score` (game score), `episode/reward_sum` (training reward), `episode/loss` (avg network loss).
- Per-interval logging: `interval_wins`, `interval_win_rate`, `interval_avg_loss` every `C` epochs.
- `wandb.finish()` called at end of training.
- Commented out `matplotlib` import and `plot_results()` function.
- Window caption shows test number during training (`Angry Birds - Test {num}`).

### Added episode scoring system
- Scoring constants in `constants.py`: `SCORE_BIRD_FIRED=-10`, `SCORE_PIG_KILLED=+25`, `SCORE_WIN=+50`, `SCORE_LOSS=-25`.
- Score distinguishes win efficiency: 1-shot win = 90, 2-shot win = 80, 10-shot loss = -125.
- Commented out `matplotlib` import and `plot_results()` function.

### Fixed reward accumulation across shot frames
- `self.reward = 0` was at top of every `move()` call, resetting reward each physics frame. Only the last frame's reward survived — shot cost, pig kills mid-flight, and block-pig collide rewards were all lost.
- Moved `self.reward = 0` inside `action is not None` block so it only resets when a new shot starts. Rewards now accumulate across all simulation frames of one shot.
- `REWARD_ALL_PIGS_DEAD` was firing on every frame after pigs were all dead. Now only fires on the frame the last pig actually dies.

### Fixed `end_of_game()` zeroing reward
- `end_of_game()` had `self.reward = 0` on both win and loss branches, wiping out terminal rewards (pig kill, all-pigs-dead, loss penalty) before they were returned. Removed.

### Added episode print summary
- Each episode prints: test num, epoch, WIN/LOSS, score, reward, loss, pigs killed, tries left, shots fired, epsilon.

### Added `pygame.event.pump()` to `render()`
- Windows marked the game window as "not responding" during training because the event queue was never drained. Added `pygame.event.pump()` in `render()`.

### Fixed DDQN implementation (critical bug)
The training loop was broken — it was **not** a correct DQN or DDQN:
- `DQN.__call__` had a custom override that **ignored the `actions` parameter** and returned all 100 Q-values. Both `Q(states, actions)` and `Q_hat(next_states, next_actions)` returned shape `[batch, 100]` instead of the Q-value for the specific action.
- The Huber loss was computed over all 100 outputs against the same broadcasted reward target, training every output toward the same value.
- `player.get_actions()` ran one-by-one inference for each sample in the batch (slow and unnecessary).

**Fix:**
- Removed the broken `DQN.__call__` override — `nn.Module.__call__` now properly calls `forward`.
- **Q(s, a)**: `Q(states)` → `gather` at the taken action index → `[batch, 1]`.
- **DDQN target**: online net picks best next action via `argmax`, target net evaluates it via `gather` → `[batch, 1]`.
- Loss now compares matching scalar Q-values per sample, as intended.
- Eliminated the slow `player.get_actions()` loop from the training step.

### Fixed infinite-episode bug (tries going negative)
- `env.move(action)` return value was ignored in `train.py` — if the bird collided with a block on the first frame (resetting `bird.move = False` immediately), `done=True` was lost and the outer loop kept firing shots.
- With tries at -1, `end_of_game()` checked `tries == 0` which was never true again, so the game ran for hundreds of shots until pigs died by sheer volume.
- **Fix:** captured `reward, done = env.move(action)` in `train.py`; added `not done` guard to inner physics loop; changed `tries == 0` to `tries <= 0` in `end_of_game()` and loss penalty check.

### Added `.gitignore`
- Ignores `__pycache__/`, `*.pyc`, `*.pyo`, `*.pth`, `saved_models/`, `wandb/`, `.env`.

### Resolved merge conflicts
Resolved merge conflicts in `DQN.py`, `Environment.py`, `Game.py`, `ai_agent.py`, `train.py` — kept local (constants.py-based) versions over hardcoded remote values.

## 2026-03-30

### Added `constants.py`
Extracted all magic numbers and hardcoded values from every source file into a single `constants.py` module, organized by category:
- Screen settings (`WIDTH`, `HEIGHT`, `FPS`)
- Colors (`BROWN`, `BLACK`, `RED`)
- Bird parameters (start position, size, out-of-bounds thresholds)
- Physics (`GRAVITY`, `VX_SCALE`, `VY_SCALE`, `ACTION_COMPONENTS`, `NUM_ACTIONS`)
- Ground, Pig, and Block defaults (sizes, angles, rotation speed, max hits)
- Level generation parameters (building count, x-range, horizontal chance)
- Reward values (shoot cost, kill bonuses, miss/loss penalties, clamp range)
- DQN network architecture (layer sizes, gamma)
- Agent epsilon-greedy parameters (start, final, decay)
- Training hyperparameters (epochs, batch size, learning rate, grad norm, etc.)

Updated files: `Bird.py`, `Block.py`, `Pig.py`, `Graphic.py`, `Human_agent.py`, `Environment.py`, `State.py`, `DQN.py`, `ai_agent.py`, `Game.py`, `train.py`, `ReplayBuffer.py` — all now use `from constants import *`.

### Added physics tables to `PHYSICS.md`
Added two reference tables at the end of `PHYSICS.md`:
- **Maximum Height** — peak height above ground for each `action[1]` value (0–9).
- **Maximum Horizontal Distance** — 10×10 grid showing how far the bird travels for every `(action[0], action[1])` combination, capped at 655 px (screen right edge).

### Rescaled human agent mouse input
Changed `Human_agent.py` so the mouse drag maps the full available screen range to all 10 action values (0–9) for both `vx` and `vy`. Previously, `vx` was capped at 3 because the bird sits only 45 px from the left edge. Now the drag-left range (0–45 px) and drag-down range (0–85 px) each map linearly to 0–9. Updated `PHYSICS.md` to reflect the new formulas.

### Rescaled bird physics to fit screen
Changed `VX_SCALE` from 5 to 2 and `VY_SCALE` from −5 to −2 in `constants.py`. Previously 52 out of 100 action combinations sent the bird off screen; now only 3 do (actions `(8,9)`, `(9,8)`, `(9,9)`). All peak heights stay on screen — the steepest launch (`action[1]=9`) reaches y=187 instead of the old y=−485. Updated tables in `PHYSICS.md`.

### Increased gravity and vertical velocity for higher arcs
Changed `GRAVITY` from 1 to 2 and `VY_SCALE` from −2 to −5. The bird now launches with more vertical speed and falls faster, producing taller, more curved parabolic arcs. Only `action[1]=9` goes above the screen (y=−85); all other launch angles stay visible. 5 out of 100 action combinations exit right. Updated `Environment.py` ballistic calculation to use `GRAVITY` constant. Updated formulas and tables in `PHYSICS.md`.

### Increased tries to 10
Changed `INITIAL_TRIES` from 3 to 10 in `constants.py`.

### Added tries counter HUD
Added `draw_tries()` method in `Environment.py` that renders "Tries: X" in the top-left corner of the screen. Called every frame during `render()`.

### Rescaled rewards and removed clamp
Removed the [−5, +5] reward clamp that was making large rewards indistinguishable. Rescaled all reward values to fit naturally in ~[−5, +5]:
- Shot cost: 1 → 0.1
- Ballistic proximity: 2 → 0.2
- Block–pig collide: 3 → 0.3
- Pig falls to ground: 50 → 1.0
- Pig killed: 100 → 3.0
- All pigs dead: 300 → 5.0
- Miss penalty: −20 → −1.0
- Loss penalty: −300 → −5.0

### Fixed reward bugs and cleaned up reward structure
- **Bug fix:** `self.reward` now resets to 0 at the start of each `move()` call. Previously it accumulated across the entire episode, leaking rewards between steps.
- **Bug fix:** `pigs_before_shot` was a local variable but checked via `hasattr(self, ...)`. Changed to `self.pigs_before_shot` so the miss penalty actually triggers.
- **Bug fix:** Miss penalty sign was inverted (`-= -1.0` added reward). Now uses `+= REWARD_MISS_PENALTY` directly.
- **Removed** `REWARD_PIG_GROUND` (+1.0) — pig kills were double-counted (ground + kill). Now only `REWARD_PIG_KILL` (+3.0) fires.
- **Removed** `REWARD_BALLISTIC_HIT` — ballistic proximity at launch time was a noisy signal that rewarded intent, not results.
