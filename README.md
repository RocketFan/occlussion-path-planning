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

Plans the path of an observer vehicle that tries to maintain line-of-sight (LOS) with a moving target vehicle while respecting speed constraints.

#### Key Design Principle: Time-Aware Sequential Planning

The problem is fundamentally **sequential in time**. Every piece of data is indexed by a specific time \( t_k = t + k \cdot dt \):

- The target is at a **specific position** \( p_\text{target}(t_k) \) at each future time step.
- Each target position produces a **different visibility polygon** \( V(t_k) \) (because the target is in a different location relative to obstacles).
- The observer's **reachable set** at \( t_k \) depends on where it was at \( t_{k-1} \): it is a disk of radius \( v_\max \cdot dt \) centred on \( \text{pos}_{k-1} \).
- Therefore: the observer position at step `k` must lie in the **intersection** of \( V(t_k) \) (for LOS) and \( \text{Reachable}(\text{pos}_{k-1}) \) (for feasibility).

All three planners respect this structure: trajectories are rolled out **step by step**, each step checks LOS against the **time-correct** target position and visibility polygon, and each step's position is **reachable from the previous step**.

Each planner also accepts a `preferred_distance` parameter (default 5.0 units).  Rather than pulling the observer as close to the target as possible, the distance objective penalises deviation from this preferred distance -- the observer tries to orbit the target at a comfortable range.  Maintaining visibility is always the **dominant** objective; the distance preference is secondary and only steers the observer when LOS permits.

#### Shared Helpers

| Function | Description |
|---|---|
| `has_line_of_sight(p1, p2, segments) -> bool` | Casts a ray from `p1` toward `p2`; if any obstacle segment is hit with `0 < t < 1`, LOS is blocked. Reuses `_ray_segment_intersection` from `visibility.py`. |
| `_move_towards(current, target, max_move) -> Point2D` | Returns a point moved from `current` toward `target` by at most `max_move` distance. Used by greedy forward simulation and as a fallback. |
| `precompute_target_visibility(target_vehicle, obstacles, dt, max_vis_radius) -> list[Polygon]` | Pre-computes the visibility polygon `V(t_k)` for each time step `t_k = 0, dt, 2*dt, ...` along the target's path. Indexed by frame: `vis_polys[k]` is the visibility polygon when the target is at `position_at(k * dt)`. |
| `_signed_distance_to_vis_boundary(point, vis_poly) -> float` | Returns positive distance if the point is **inside** the visibility polygon (has LOS), negative if **outside**. Provides a continuous, smooth metric for optimisation (better than binary LOS). |
| `plan_observer_path(strategy, target_vehicle, obstacles, start_pos, max_speed, dt, max_vis_radius) -> list[Point2D]` | Main planning loop. Pre-computes obstacle segments and time-indexed visibility polygons, then iterates over time steps passing the correct time index and current observer position to `strategy.step()`. Returns the full observer trajectory. |

#### Strategy 1: Greedy with Lookahead (`GreedyLookaheadPlanner`)

At each time step, sample candidate next positions within the reachable set, score each by **sequentially simulating forward** over the horizon, and pick the best.

**Algorithm at time `t` with observer at `pos`:**

1. **Sample candidates:** Generate `num_candidates` positions on a circle of radius `max_speed * dt` around `pos`, plus "stay" at `pos`.
2. **Score each candidate via forward simulation.** For candidate `c` at time `t + dt`:
   - Set `sim_pos = c`, `score = 0`.
   - For `k = 1, 2, ..., horizon_steps`:
     - Compute `t_k = t + k * dt` and the target position at that time.
     - Check: is `sim_pos` inside `vis_polys[frame + k]`?
     - If yes: `score += 1` (LOS maintained).
     - Greedy propagation: if LOS, stay; if no LOS, move `sim_pos` toward `target_k` by `max_speed * dt`.
     - **Distance preference:** `score += 0.05 / (1 + |dist_to_target - preferred_distance|)` (small bonus that peaks when the observer is at the preferred distance; secondary to the integer LOS score).
   - **Visibility bonus:** `score += signed_distance(c, vis_polys[frame + 1]) * 0.01` (prefer being deep inside the visibility polygon, not on the edge).
3. **Pick** the candidate with the highest score.

The forward simulation is fully sequential -- each simulated step starts from the previous simulated position and checks LOS against the time-correct visibility polygon. Reachability is enforced at every step.

**Parameters:** `num_candidates=36`, `horizon_steps=10`, `preferred_distance=5.0`

#### Strategy 2: MPPI (`MPPIPlanner`)

Sample `K` **entire trajectories** (sequences of velocity vectors) over the horizon. Each trajectory is a time-indexed sequence of positions rolled out step-by-step from the current observer position. Score each trajectory and compute the next action as a soft exponentially-weighted average.

**Algorithm at time `t` with observer at `pos`:**

1. **Maintain nominal control sequence** `U = [u_0, ..., u_{H-1}]` (2-D velocity vectors, warm-started from the shifted previous solution).
2. **Sample `K` perturbations:** `U_k = U + noise_k`, where `noise_k ~ N(0, sigma^2 * I)`. Clamp each `||u_i|| <= max_speed`.
3. **Roll out each trajectory sequentially:**
   - `pos_0 = pos` (current observer position).
   - For `i = 0, ..., H-1`:
     - `pos_{i+1} = pos_i + U_k[i] * dt` (each position depends on the previous).
     - `t_i = t + (i+1) * dt`.
     - Check LOS: `has_line_of_sight(pos_{i+1}, target_at(t_i), segments)`.
4. **Score each trajectory:** `S_k = sum(w_los * los_i + w_dist / (1 + |dist(pos_i, target_i) - preferred_distance|))` where each `target_i` is at the time-correct position. The distance reward peaks when the observer is at `preferred_distance` from the target.
5. **Compute weights:** `w_k = exp(S_k / lambda)`, normalise.
6. **Update nominal sequence:** `U = sum(w_k * U_k)`.
7. **Apply first action:** `next_pos = pos + U[0] * dt`, clamped to `max_speed`.
8. **Shift** for warm start: `U = [U[1], ..., U[H-1], 0]`.

Every trajectory is a time-indexed sequence. Position at step `i` is derived from step `i-1` (reachability). LOS at step `i` is checked against the time-correct target. The soft averaging naturally balances exploration.

**Parameters:** `K=128`, `horizon_steps=10`, `sigma=1.5`, `lambda_=0.1`, `preferred_distance=5.0`

#### Strategy 3: Scipy Optimisation MPC (`ScipyMPCPlanner`)

Optimise a differentiable cost function over the full **time-indexed** horizon trajectory. Reachability between consecutive steps is enforced via penalty. LOS quality is measured via signed distance to the time-correct visibility polygon boundary.

**Decision variable:** flattened trajectory `[x_1, y_1, ..., x_H, y_H]` where `(x_k, y_k)` is the observer position at time `t + k * dt`.

**Cost function `J(trajectory)`:**

- **LOS term** (smooth, time-indexed): for each step `k`, compute the signed distance from `(x_k, y_k)` to the boundary of `vis_polys[frame + k]`. Penalise being outside: `w_los * sum_k( max(0, -signed_dist_k)^2 )`.
- **Distance-preference term** (time-indexed): `w_dist * sum_k( (||pos_k - target_k|| - preferred_distance)^2 )`. Each step penalises deviation from `preferred_distance` to where the target actually is at that time, creating a "ring" attractor rather than a point attractor.
- **Smoothness term**: `w_smooth * sum_k( ||(pos_{k+1} - 2*pos_k + pos_{k-1})||^2 )` -- penalises jerky motion (acceleration penalty).
- **Speed limit** (sequential reachability): `w_speed * sum_k( max(0, ||pos_k - pos_{k-1}|| - max_speed*dt)^2 )`. Enforces that each step is reachable from the previous step within the speed limit.

**Solver:** `scipy.optimize.minimize(method='L-BFGS-B')`, warm-started from the shifted previous solution. After solving, apply first position `(x_1, y_1)` and shift the trajectory for warm start at the next time step (receding horizon).

**Parameters:** `horizon_steps=10`, `w_los=10.0`, `w_dist=1.0`, `w_smooth=0.5`, `w_speed=100.0`, `preferred_distance=5.0`

#### Complexity Estimates

- **Greedy Lookahead:** 36 candidates x 10 horizon steps x ~30 segments = ~10K LOS checks per planning step. ~1.5M total over ~140 steps.
- **MPPI:** 128 trajectories x 10 steps x ~30 segments = ~38K per planning step. ~5.4M total.
- **Scipy MPC:** ~50 optimiser iterations x 10 steps x 1 vis polygon containment check = ~500 per step, but vis polygon distance is more expensive than raw ray checks. ~15K equivalent per step. ~2.1M total.
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
| `animate_two_vehicles(fig, ax, map_env, target_vehicle, observer_positions, ...)` | Returns a `FuncAnimation` showing both target (orange) and observer (cyan) vehicles. Draws the target's visibility polygon (yellow), predicted path (green dashed), and a LOS ray between the two vehicles -- solid green when clear, dashed red when occluded. Observer positions are used directly (one per frame) for synchronisation with the planner output. |

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
