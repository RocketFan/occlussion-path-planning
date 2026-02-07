# Occlusion-Aware Path Planning

A modular Python pipeline that builds a 2-D map with polygon obstacles, discretises it into a grid, finds an A\* path from start to goal, computes 2-D visibility polygons along the path, and animates a vehicle traversing the route with real-time visibility and path prediction.

## Architecture

```
map_environment.py ──► pathfinding.py ──► visualization.py ──► notebook.ipynb
        │                                        ▲
        └──────────► visibility.py ──────────────┘
                                                  ▲
               vehicle.py ───────────────────────┘
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
| `compute_visibility_polygon(observer, obstacles, max_radius) -> Polygon` | Casts rays toward every obstacle vertex (± ε for corners) and 64 uniformly-spaced angles. Finds closest intersection per ray, builds a `shapely.Polygon`, clips to a bounding circle. Collapses `MultiPolygon` results to the largest component. |
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

## Notebook (`notebook.ipynb`)

The notebook orchestrates the full pipeline in six steps:

1. **Create Map** — `create_sample_map()`, visualised with `plot_map()`.
2. **Grid Overlay** — `create_grid()`, visualised with `plot_grid()`.
3. **A\* Pathfinding** — defines start `(1, 1)` and goal `(28, 28)`, runs `find_grid_path()`, visualised with `plot_path()`.
4. **Visibility at Key Waypoints** — computes visibility at 5 representative waypoints (start, ¼, ½, ¾, end), shown in a 2×3 subplot grid.
5. **Combined Visibility** — `compute_path_visibility()` for the entire path, overlaid on a single plot.
6. **Animated Vehicle Traversal** — a `Vehicle` at speed 2.0 units/s follows the path. The animation shows position, heading, 3-second path prediction (8 sample dots), and real-time visibility.

> The notebook uses the `%matplotlib widget` backend (`ipympl`) for interactive animation.

## Dependencies

Listed in `requirements.txt`:

```
matplotlib
shapely
numpy
astar
ipympl
```

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Open notebook.ipynb in Jupyter / VS Code and run all cells
```
