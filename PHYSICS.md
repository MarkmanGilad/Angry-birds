# Bird Physics & Action Space

## Gravity

The bird obeys simple discrete kinematics. Every frame:

```
x  += vx        # constant horizontal speed
y  += vy        # vertical position
vy += 2         # gravity: g = 2 px/frame²
```

The y-axis is **inverted** (pygame convention): positive y points **down**, so negative `vy` means the bird moves **upward**.

---

## Action Space

An action is a tuple `(action[0], action[1])` where each component is an integer in `[0, 9]`, giving a 10×10 = **100 possible actions**.

### Mapping to initial velocity

| Component | Formula | action=0 | action=1 | action=2 | action=3 | action=4 | action=5 | action=6 | action=7 | action=8 | action=9 |
|-----------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| `vx` | `(action[0] + 1) × 2` | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
| `vy₀` | `(action[1] - 1) × (−5)` | +5 ↓ | 0 | −5 ↑ | −10 ↑ | −15 ↑ | −20 ↑ | −25 ↑ | −30 ↑ | −35 ↑ | −40 ↑ |

- `action[0]` controls **horizontal speed** — higher = faster, farther range.
- `action[1]` controls **launch angle** — low values shoot downward, high values shoot steeply upward.

---

## Trajectory

Bird starts at `(x₀, y₀) = (45, 315)`.

$$x(t) = 45 + vx \cdot t$$

$$y(t) = 315 + vy_0 \cdot t + \frac{g \cdot t^2}{2} \quad (g = 2)$$

### Peak height

Reached when `vy = 0`, i.e. at frame $t_{peak} = -vy_0 / g$:

$$y_{peak} = 315 - \frac{vy_0^2}{2g} = 315 - \frac{vy_0^2}{4}$$

Example: `action[1]=9` → `vy₀=-40` → $y_{peak} = 315 - 400 = -85$ (just above screen top, rare).

### Time to reach a target x

$$t = \frac{x_{target} - 45}{vx}$$

### Screen exit conditions

The bird is removed when:
- `y > 400` (fell below ground, floor is at y=360)
- `x > 700` (flew past right edge of screen)

---

## Human Agent Controls

The human controls the bird via a slingshot-style mouse drag from the bird position `(45, 315)`:

- **Drag left** → higher `action[0]` (faster horizontal speed)
- **Drag down** → higher `action[1]` (steeper upward launch)

```python
dx = 45 - mouse_x          # positive when dragging left
dy = mouse_y - 315         # positive when dragging down
vx = int(dx * 9 / 45)      # maps full left drag (0–45 px) to action 0–9
vy = int(dy * 9 / 85)      # maps full down drag (0–85 px) to action 0–9
# both clamped to [0, 9]
```

All 10 values of both `action[0]` and `action[1]` are reachable via mouse drag.

---

## DQN Action Encoding

Actions are flattened to a single index for the neural network:

```python
index = action[0] * 10 + action[1]   # 0 .. 99
action = (index // 10, index % 10)   # reverse
```

The network outputs **100 Q-values** (one per action pair) and picks `argmax`.

---

## Maximum Reach Tables

### Maximum Height (by `action[1]`)

Height above ground at the trajectory peak. Ground is at y = 360, bird launches from y = 315.

$$\text{height} = 360 - y_{peak} = 45 + \frac{vy_0^2}{2g} = 45 + \frac{vy_0^2}{4}$$

| `action[1]` | `vy₀` | Peak y (px) | Height above ground (px) |
|:---:|:---:|:---:|:---:|
| 0 | +5 ↓ | 315 | 45 |
| 1 | 0 | 315 | 45 |
| 2 | −5 ↑ | 308.8 | 51.2 |
| 3 | −10 ↑ | 290 | 70 |
| 4 | −15 ↑ | 258.8 | 101.2 |
| 5 | −20 ↑ | 215 | 145 |
| 6 | −25 ↑ | 158.8 | 201.2 |
| 7 | −30 ↑ | 90 | 270 |
| 8 | −35 ↑ | 8.8 | 351.2 |
| 9 | −40 ↑ | −85 | 445 |

> Only `action[1]=9` sends the bird above the visible screen (y < 0). `action[1]=8` barely stays on screen.

### Maximum Horizontal Distance (`action[0]` × `action[1]`)

Distance from launch point in pixels. Bird starts at x = 45; capped at **655 px** (right edge x = 700).

| `vx` ╲ `vy₀` | +5 ↓ | 0 | −5 ↑ | −10 ↑ | −15 ↑ | −20 ↑ | −25 ↑ | −30 ↑ | −35 ↑ | −40 ↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2** | 14 | 18 | 24 | 31 | 39 | 47 | 56 | 65 | 75 | 84 |
| **4** | 28 | 37 | 48 | 62 | 78 | 94 | 112 | 130 | 149 | 168 |
| **6** | 42 | 55 | 72 | 93 | 116 | 142 | 168 | 196 | 224 | 252 |
| **8** | 56 | 74 | 96 | 124 | 155 | 189 | 224 | 261 | 298 | 336 |
| **10** | 71 | 92 | 121 | 155 | 194 | 236 | 280 | 326 | 373 | 420 |
| **12** | 85 | 111 | 145 | 186 | 233 | 283 | 336 | 391 | 447 | 504 |
| **14** | 99 | 129 | 169 | 217 | 271 | 330 | 392 | 456 | 522 | 588 |
| **16** | 113 | 148 | 193 | 248 | 310 | 378 | 449 | 522 | 596 | **655** |
| **18** | 127 | 166 | 217 | 279 | 349 | 425 | 505 | 587 | **655** | **655** |
| **20** | 141 | 184 | 241 | 310 | 388 | 472 | 561 | 652 | **655** | **655** |

> Only **5 out of 100** action combinations send the bird off screen: `(7,9)`, `(8,8)`, `(8,9)`, `(9,8)`, `(9,9)`.
