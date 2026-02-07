# Occlusion-Aware Path Planning

A modular Python pipeline that builds a 2-D map with polygon obstacles, discretises it into a grid, finds an A\* path from start to goal, computes 2-D visibility polygons along the path, and animates a vehicle traversing the route with real-time visibility and path prediction. An **observer vehicle** can then be planned to follow the target while maintaining line-of-sight, using one of three MPC-style strategies.

## Architecture

```
map_environment.py ──► pathfinding.py ──► visualization.py ──► notebook.ipynb
        │                                        ▲
        └──────────► visibility.py ──────────────┘
                          ▲                       ▲
               vehicle.py ┘                       │
                    ▲                              │
                    └── observer_planner.py ───────┘
```

## Modules

### `map_environment.py` — Map and Grid

Defines the core data types and the map environment.

| Symbol | Description |
|---|---|
| `Point2D(NamedTuple)` | A point in continuous 2-D space (`x: float`, `y: float`). |
| `GridCell` | Type alias `tuple[int, int]` representing a grid cell `(row, col)`. |
| `MapEnvironment` | Rectangular map with polygon obstacles and a cached occupancy grid. |
| `create_sample_map() -> MapEnvironment` | Factory that returns a 30×30 map with six polygon obstacles (L-shape, rectangle, triangle, wall, square, quadrilateral). |

Key `MapEnvironment` methods:

- `create_grid() -> npt.NDArray[np.bool_]` — builds/returns a boolean grid (`True` = free, `False` = blocked).
- `is_free(row: int, col: int) -> bool` — queries the grid.
- `cell_center(row: int, col: int) -> Point2D` — continuous centre of a cell.
- `xy_to_cell(point: Point2D) -> GridCell` — converts continuous coords to grid cell.
- `grid_rows: int`, `grid_cols: int` — property accessors.

### `pathfinding.py` — A\* on the Grid

Uses the [`astar`](https://github.com/jrialland/python-astar) library.

| Symbol | Description |
|---|---|
| `GridAStar(AStar[GridCell])` | A\* solver on an 8-connected occupancy grid. Cardinal cost = 1.0, diagonal cost = √2. Heuristic: Euclidean distance. |
| `find_grid_path(map_env, start, goal) -> list[Point2D] \| None` | Converts continuous start/goal to grid cells, runs A\*, and returns the path as a list of cell-centre `Point2D` coordinates. Returns `None` if no path exists. |

### `visibility.py` — 2-D Visibility Polygon

Implements a ray-casting visibility algorithm.

| Symbol | Description |
|---|---|
| `compute_visibility_polygon(observer, obstacles, max_radius) -> Polygon` | Casts rays toward every obstacle vertex (± ε for corners) and 360 uniformly-spaced angles. Finds closest intersection per ray, builds a `shapely.Polygon`, clips to a bounding circle. Collapses `MultiPolygon` results to the largest component. |
| `compute_path_visibility(path, obstacles, max_radius) -> list[Polygon]` | Runs `compute_visibility_polygon` for every point on the path. |

### `vehicle.py` — Vehicle Simulation

| Symbol | Description |
|---|---|
| `Vehicle(path, speed)` | A point vehicle that linearly interpolates along a pre-planned path at a given speed. Pre-computes cumulative arc-length distances for efficient queries. |

Key `Vehicle` methods / properties:

- `position_at(t: float) -> Point2D` — interpolated position at time `t` (binary search over segments).
- `heading_at(t: float) -> float` — heading angle in radians at time `t`.
- `step(dt: float) -> Point2D` — advances elapsed time and returns the new position.
- `predict_positions(t, horizon, num_samples) -> list[Point2D]` — samples `num_samples` evenly-spaced future positions over the next `horizon` seconds from time `t`. Stops early if the vehicle reaches the end of the path.
- `total_length`, `total_time`, `elapsed`, `finished`, `reset()`.

### `observer_planner.py` — Observer Vehicle Planning

Plans the path of an observer vehicle that tries to maintain line-of-sight (LOS) with a moving target vehicle using **Dubins vehicle dynamics**.

#### Dubins Dynamics

The observer vehicle state is \( (x, y, \theta) \) where \( \theta \) is the heading angle. The vehicle moves at a **constant forward speed** and the only control input is the **turn rate** \( \omega \) (rad/s), bounded by \( \omega_{\max} = \text{speed} / \text{min\_turning\_radius} \). With default parameters (`speed = 3.0`, `min_turning_radius = 2.5`) this gives \( \omega_{\max} = 1.2 \) rad/s.

At each time step `dt`, given turn rate \( \omega \):

- **Arc motion** (\( |\omega| > \epsilon \)):
  - \( x' = x + \frac{v}{\omega}[\sin(\theta + \omega \cdot dt) - \sin(\theta)] \)
  - \( y' = y + \frac{v}{\omega}[-\cos(\theta + \omega \cdot dt) + \cos(\theta)] \)
- **Straight-line** (\( \omega \approx 0 \)):
  - \( x' = x + v \cos(\theta) \cdot dt \)
  - \( y' = y + v \sin(\theta) \cdot dt \)
- \( \theta' = \theta + \omega \cdot dt \)

#### Key Design Principle: Time-Aware Sequential Planning

The problem is fundamentally **sequential in time**. Every piece of data is indexed by a specific time \( t_k = t + k \cdot dt \):

- The target is at a **specific position** \( p_\text{target}(t_k) \) at each future time step.
- Each target position produces a **different visibility polygon** \( V(t_k) \) (because the target is in a different location relative to obstacles).
- The observer's **reachable set** at \( t_k \) depends on where it was at \( t_{k-1} \): it is determined by the Dubins dynamics (constant speed, bounded turn rate) from the previous pose \( (x_{k-1}, y_{k-1}, \theta_{k-1}) \).
- Therefore: the observer pose at step `k` must satisfy both \( V(t_k) \) (for LOS) and Dubins feasibility from the previous pose.

All three planners respect this structure: trajectories are rolled out **step by step** using `_dubins_step`, each step checks LOS against the **time-correct** target position and visibility polygon, and each step's pose is **reachable from the previous step** under Dubins constraints.

Each planner also accepts a `preferred_distance` parameter (default 5.0 units).  Rather than pulling the observer as close to the target as possible, the distance objective penalises deviation from this preferred distance -- the observer tries to orbit the target at a comfortable range.  Maintaining visibility is always the **dominant** objective; the distance preference is secondary and only steers the observer when LOS permits.

#### Shared Helpers

| Function | Description |
|---|---|
| `has_line_of_sight(p1, p2, segments) -> bool` | Casts a ray from `p1` toward `p2`; if any obstacle segment is hit with `0 < t < 1`, LOS is blocked. Reuses `_ray_segment_intersection` from `visibility.py`. |
| `_dubins_step(x, y, theta, speed, omega, dt) -> (x', y', theta')` | Advances one Dubins step using exact circular-arc integration. Takes the turn rate `omega` directly (not normalised). |
| `_dubins_steer_towards(x, y, theta, target_x, target_y, speed, dt, omega_max) -> (x', y', theta')` | Picks the turn rate `omega` that best steers toward a target point, clamped to `[-omega_max, omega_max]`. Used as the greedy fallback during forward simulation. |
| `precompute_target_visibility(target_vehicle, obstacles, dt, max_vis_radius) -> list[Polygon]` | Pre-computes the visibility polygon `V(t_k)` for each time step `t_k = 0, dt, 2*dt, ...` along the target's path. Indexed by frame: `vis_polys[k]` is the visibility polygon when the target is at `position_at(k * dt)`. |
| `_signed_distance_to_vis_boundary(point, vis_poly) -> float` | Returns positive distance if the point is **inside** the visibility polygon (has LOS), negative if **outside**. Provides a continuous, smooth metric for optimisation (better than binary LOS). |
| `plan_observer_path(strategy, target_vehicle, obstacles, start_pos, start_heading, speed, dt, max_vis_radius) -> (list[Point2D], list[float])` | Main planning loop. Pre-computes obstacle segments and time-indexed visibility polygons, then iterates over time steps passing the correct time index and current observer pose to `strategy.step()`. Returns `(positions, headings)`. |

#### Strategy 1: Greedy with Lookahead (`GreedyLookaheadPlanner`)

At each time step, sample candidate turn rates within `[-omega_max, omega_max]`, score each by **sequentially simulating forward** over the horizon using Dubins dynamics, and pick the best.

**Algorithm at time `t` with observer at `(x, y, theta)`:**

1. **Sample candidates:** Generate `num_candidates` turn rates uniformly from `[-omega_max, omega_max]`, plus `omega = 0` (straight).
2. **Score each candidate via forward simulation.** For candidate turn rate `omega_c`:
   - Simulate one Dubins step to get `(sx, sy, stheta)`.
   - Set `sim_x, sim_y, sim_theta = sx, sy, stheta`, `score = 0`.
   - For `k = 1, 2, ..., horizon_steps`:
     - Compute `t_k = t + k * dt` and the target position at that time.
     - Check: is `(sim_x, sim_y)` inside `vis_polys[frame + k]`?
     - If yes: `score += 1` (LOS maintained); continue straight (`omega = 0`).
     - If no: greedily steer toward target via `_dubins_steer_towards`.
     - **Distance preference:** `score += 0.05 / (1 + |dist_to_target - preferred_distance|)`.
   - **Visibility bonus:** `score += signed_distance((sx, sy), vis_polys[frame + 1]) * 0.01`.
3. **Pick** the candidate with the highest score; return its first-step pose.

**Parameters:** `num_candidates=36`, `horizon_steps=10`, `preferred_distance=5.0`, `min_turning_radius=2.5`

#### Strategy 2: MPPI (`MPPIPlanner`)

Sample `K` **entire trajectories** (sequences of scalar turn rates) over the horizon. Each trajectory is a time-indexed sequence of poses rolled out step-by-step from the current observer pose using Dubins dynamics. Score each trajectory and compute the next action as a soft exponentially-weighted average.

**Algorithm at time `t` with observer at `(x, y, theta)`:**

1. **Maintain nominal control sequence** `U = [omega_0, ..., omega_{H-1}]` (scalar turn rates, warm-started from the shifted previous solution).
2. **Sample `K` perturbations:** `U_k = U + noise_k`, where `noise_k ~ N(0, sigma^2)`. Clamp each `omega_i` to `[-omega_max, omega_max]`.
3. **Roll out each trajectory sequentially** using `_dubins_step`:
   - `pose_0 = (x, y, theta)`.
   - For `i = 0, ..., H-1`:
     - `pose_{i+1} = _dubins_step(pose_i, speed, U_k[i], dt)`.
     - `t_i = t + (i+1) * dt`.
     - Check LOS: `has_line_of_sight(pos_{i+1}, target_at(t_i), segments)`.
4. **Score each trajectory:** `S_k = sum(w_los * los_i + w_dist / (1 + |dist(pos_i, target_i) - preferred_distance|))`.
5. **Compute weights:** `w_k = exp(S_k / lambda)`, normalise.
6. **Update nominal sequence:** `U = sum(w_k * U_k)`.
7. **Apply first action:** simulate one Dubins step with `omega_0`, clamped to `omega_max`.
8. **Shift** for warm start: `U = [U[1], ..., U[H-1], 0]`.

**Parameters:** `K=128`, `horizon_steps=10`, `sigma=1.5`, `lambda_=0.1`, `preferred_distance=5.0`, `min_turning_radius=2.5`

#### Strategy 3: Scipy Optimisation MPC (`ScipyMPCPlanner`)

Optimise a differentiable cost function over a **turn-rate sequence** using Dubins dynamics. LOS quality is measured via signed distance to the time-correct visibility polygon boundary.

**Decision variable:** turn-rate sequence `[omega_1, ..., omega_H]` where `omega_k` is the turn rate applied at step `k`.  The trajectory is reconstructed from the current pose `(x, y, theta)` by forward-simulating Dubins dynamics with each turn rate.

**Cost function `J(omega_sequence)`:**

- **LOS term** (smooth, time-indexed): for each step `k`, forward-simulate from the current pose using the turn-rate sequence, then compute the signed distance from the resulting position to the boundary of `vis_polys[frame + k]`. Penalise being outside: `w_los * sum_k( max(0, -signed_dist_k)^2 )`.
- **Distance-preference term** (time-indexed): `w_dist * sum_k( (||pos_k - target_k|| - preferred_distance)^2 )`. Each step penalises deviation from `preferred_distance` to where the target actually is at that time.
- **Smoothness term** (turn-rate jerk): `w_smooth * sum_k( |omega_{k+1} - omega_k|^2 )` -- penalises abrupt changes in turn rate.

**Bounds:** `[(-omega_max, omega_max)] * H` -- box bounds on each turn rate. No constraint function is needed because the constant speed is implicit in the Dubins dynamics.

**Solver:** `scipy.optimize.minimize(method='L-BFGS-B')` with box bounds on omega, warm-started from the shifted previous solution. After solving, apply the first turn rate via `_dubins_step` and shift the sequence for warm start at the next time step (receding horizon).

**Parameters:** `horizon_steps=10`, `w_los=10.0`, `w_dist=1.0`, `w_smooth=0.5`, `preferred_distance=5.0`, `min_turning_radius=2.5`

#### Complexity Estimates

- **Greedy Lookahead:** 36 candidate turn rates x 10 horizon steps x ~30 segments = ~10K LOS checks per planning step. ~1.5M total over ~140 steps.
- **MPPI:** 128 trajectories x 10 Dubins steps x ~30 segments = ~38K per planning step. ~5.4M total.
- **Scipy MPC:** ~50 optimiser iterations x 10 Dubins steps x 1 vis polygon containment check = ~500 per step, but vis polygon distance is more expensive than raw ray checks. ~15K equivalent per step. ~2.1M total.
- **Precomputed visibility polygons:** ~140 computations (one per dt step), each ~400 ray casts x 30 segments = ~12K. Total: ~1.7M (one-time cost, shared by all planners).

All comfortably fast for offline planning in Python (< 10s each).

### `visualization.py` — Plotting and Animation

All static plot functions take a `matplotlib.axes.Axes` and return `None`:

| Function | Description |
|---|---|
| `plot_map(ax, map_env)` | Draws map boundary and filled obstacle polygons. |
| `plot_grid(ax, map_env)` | Overlays the occupancy grid (semi-transparent). |
| `plot_path(ax, path)` | Draws the A\* path with start/goal markers. |
| `plot_visibility(ax, observer, vis_poly)` | Draws a visibility polygon (handles both `Polygon` and `MultiPolygon`). |
| `plot_all_visibility(ax, path, vis_polys)` | Draws visibility for every waypoint. |

Animation:

| Function | Description |
|---|---|
| `animate_vehicle(fig, ax, map_env, vehicle, ...)` | Returns a `FuncAnimation` that shows the vehicle (orange dot + heading line), predicted future positions (green dashed line + dots), and a real-time visibility polygon (yellow region) updating each frame. Key parameters: `max_vis_radius`, `prediction_horizon`, `prediction_samples`, `dt`, `interval_ms`. |
| `animate_two_vehicles(fig, ax, map_env, target_vehicle, observer_positions, observer_headings, ...)` | Returns a `FuncAnimation` showing both target (orange) and observer (cyan) vehicles. Draws the target's visibility polygon (yellow), predicted path (green dashed), and a LOS ray between the two vehicles -- solid green when clear, dashed red when occluded. Observer positions and headings are used directly (one per frame) for synchronisation with the planner output. |

## Notebook (`notebook.ipynb`)

The notebook orchestrates the full pipeline in nine steps:

1. **Create Map** — `create_sample_map()`, visualised with `plot_map()`.
2. **Grid Overlay** — `create_grid()`, visualised with `plot_grid()`.
3. **A\* Pathfinding** — defines start `(1, 1)` and goal `(28, 28)`, runs `find_grid_path()`, visualised with `plot_path()`.
4. **Visibility at Key Waypoints** — computes visibility at 5 representative waypoints (start, 1/4, 1/2, 3/4, end), shown in a 2x3 subplot grid.
5. **Combined Visibility** — `compute_path_visibility()` for the entire path, overlaid on a single plot.
6. **Animated Vehicle Traversal** — a `Vehicle` at speed 2.0 units/s follows the path. The animation shows position, heading, 3-second path prediction (8 sample dots), and real-time visibility.
7. **Observer: Greedy Lookahead** — plans an observer vehicle path using `GreedyLookaheadPlanner`, reports LOS maintenance percentage, and animates both vehicles with `animate_two_vehicles`.
8. **Observer: MPPI** — same as Step 7, using `MPPIPlanner`.
9. **Observer: Scipy MPC** — same as Step 7, using `ScipyMPCPlanner`.

> The notebook uses the `%matplotlib widget` backend (`ipympl`) for interactive animation.

## Dependencies

Listed in `requirements.txt`:

```
matplotlib
shapely
numpy
astar
ipympl
scipy
```

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Open notebook.ipynb in Jupyter / VS Code and run all cells
```
