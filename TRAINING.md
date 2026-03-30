# Training — State & Action Reference

## State Vector

The state is a flat tensor of size **17** (`1 + 2×MAX_PIGS + 6×MAX_BLOCKS`).

| Index | Feature | Normalization |
|:---:|---|---|
| 0 | `tries` (remaining shots) | raw integer |
| 1 | Pig 1 center x | ÷ 700 (WIDTH) |
| 2 | Pig 1 center y | ÷ 400 (HEIGHT) |
| 3 | Pig 2 center x | ÷ 700 |
| 4 | Pig 2 center y | ÷ 400 |
| 5 | Block 1 bottom-left x | ÷ 700 |
| 6 | Block 1 bottom-left y | ÷ 400 |
| 7 | Block 1 width | ÷ 100 |
| 8 | Block 1 height | ÷ 100 |
| 9 | Block 1 angle | ÷ 360 |
| 10 | Block 1 hit count | ÷ 2 |
| 11 | Block 2 bottom-left x | ÷ 700 |
| 12 | Block 2 bottom-left y | ÷ 400 |
| 13 | Block 2 width | ÷ 100 |
| 14 | Block 2 height | ÷ 100 |
| 15 | Block 2 angle | ÷ 360 |
| 16 | Block 2 hit count | ÷ 2 |

- If fewer than `MAX_PIGS` or `MAX_BLOCKS` exist, remaining slots are zero-padded.
- Positions are normalized to [0, 1] relative to screen dimensions.
- `tries` is **not** normalized (passed as raw integer).

---

## Action Space

An action is a pair `(action[0], action[1])`, each in `[0, 9]` → **100 possible actions**.

| Component | Controls | Formula | Range |
|---|---|---|---|
| `action[0]` | Horizontal speed | `vx = (action[0] + 1) × 2` | 2 – 20 px/frame |
| `action[1]` | Launch angle | `vy = (action[1] - 1) × (−5)` | +5 (down) to −40 (steep up) |

### DQN Encoding

The network outputs 100 Q-values (one per action pair). The mapping between flat index and action pair:

```
index  = action[0] × 10 + action[1]      # encode
action = (index // 10, index % 10)        # decode
```

### Epsilon-Greedy Exploration

During training, with probability ε the agent picks a random action:

$$\varepsilon = 0.01 + (1 - 0.01) \times e^{-\text{epoch} / 200000}$$

- Starts at ~1.0 (fully random)
- Decays exponentially toward 0.01

---

## Reward Structure

| Event | Reward |
|---|---|
| Firing a shot | −0.1 |
| Bird–block collision near pig | +0.3 |
| Pig killed (per pig, end of frame) | +3.0 |
| All pigs killed | +5.0 |
| Shot misses (bird exits screen, no kill) | −1.0 |
| Game over with pigs remaining | −5.0 |

Reward is reset to 0 at the start of each new shot (`action is not None`), so rewards accumulate across all physics frames of one shot.

---

## Episode Score (wandb only, not used for training)

| Event | Points |
|---|---|
| Bird fired | −10 |
| Pig killed | +25 |
| Win bonus | +50 |
| Loss penalty | −25 |

**Examples:**

| Scenario | Shots | Pigs | Score |
|---|---|---|---|
| 1-shot win (2 pigs) | 1 | 2 | −10 + 50 + 50 = **90** |
| 2-shot win (2 pigs) | 2 | 2 | −20 + 50 + 50 = **80** |
| 3-shot win (2 pigs) | 3 | 2 | −30 + 50 + 50 = **70** |
| 10-shot loss (0 pigs) | 10 | 0 | −100 − 25 = **−125** |
| 5-shot loss (1 pig) | 5 | 1 | −50 + 25 − 25 = **−50** |

---

## Network Architecture

```
Input(17) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(100)
```

- Loss: Huber (Smooth L1)
- Optimizer: Adam (lr = 0.00001)
- Gradient clipping: max norm 1.0
- Target network updated every 500 epochs
- Replay buffer: 10,000 transitions, batch size 128

---

## Training Loop

1. Reset environment, get initial state tensor
2. Agent selects action (ε-greedy)
3. Execute action, wait for physics to stabilize
4. Store `(state, action, reward, next_state, done)` in replay buffer
5. Sample batch, compute Q-targets via Bellman equation with target network
6. Backprop Huber loss, clip gradients, update weights
7. Every 500 epochs: sync target network, log win rate and average loss
8. Save model when win rate > 95% of interval, or at end of training
