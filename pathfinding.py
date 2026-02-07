"""A* pathfinding on a discretised grid using the ``astar`` library."""

from __future__ import annotations

import math
from typing import Iterable

from astar import AStar

from map_environment import GridCell, MapEnvironment, Point2D

# Eight-connected direction offsets (row_delta, col_delta)
_DIRECTIONS: list[GridCell] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_SQRT2: float = math.sqrt(2.0)


class GridAStar(AStar[GridCell]):
    """A* solver operating on an 8-connected occupancy grid."""

    def __init__(self, map_env: MapEnvironment) -> None:
        self._map_env: MapEnvironment = map_env
        # Ensure the grid is built before we start querying it.
        self._map_env.create_grid()

    # -- AStar interface ---------------------------------------------------

    def neighbors(self, node: GridCell) -> Iterable[GridCell]:
        """Yield free 8-connected neighbours of *node*."""
        row: int = node[0]
        col: int = node[1]
        for dr, dc in _DIRECTIONS:
            nr: int = row + dr
            nc: int = col + dc
            if self._map_env.is_free(nr, nc):
                yield (nr, nc)

    def distance_between(self, n1: GridCell, n2: GridCell) -> float:
        """Return 1.0 for cardinal moves, sqrt(2) for diagonal moves."""
        dr: int = abs(n1[0] - n2[0])
        dc: int = abs(n1[1] - n2[1])
        if dr + dc == 2:  # diagonal
            return _SQRT2
        return 1.0

    def heuristic_cost_estimate(self, current: GridCell, goal: GridCell) -> float:
        """Euclidean distance heuristic."""
        dr: float = float(current[0] - goal[0])
        dc: float = float(current[1] - goal[1])
        return math.hypot(dr, dc)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def find_grid_path(
    map_env: MapEnvironment,
    start: Point2D,
    goal: Point2D,
) -> list[Point2D] | None:
    """Find an A* path from *start* to *goal* on the map's grid.

    Returns a list of continuous ``Point2D`` coordinates (cell centres),
    or ``None`` when no path exists.
    """
    start_cell: GridCell = map_env.xy_to_cell(start)
    goal_cell: GridCell = map_env.xy_to_cell(goal)

    if not map_env.is_free(*start_cell):
        raise ValueError(f"Start position {start} falls inside an obstacle (cell {start_cell}).")
    if not map_env.is_free(*goal_cell):
        raise ValueError(f"Goal position {goal} falls inside an obstacle (cell {goal_cell}).")

    solver: GridAStar = GridAStar(map_env)
    result = solver.astar(start_cell, goal_cell)

    if result is None:
        return None

    path: list[Point2D] = [map_env.cell_center(r, c) for r, c in result]
    return path
