"""Map environment with polygon obstacles and occupancy grid."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon, box


class Point2D(NamedTuple):
    """A point in continuous 2-D space."""

    x: float
    y: float


GridCell = tuple[int, int]  # (row, col)


class MapEnvironment:
    """Rectangular 2-D map with polygonal obstacles and a discretised grid."""

    def __init__(
        self,
        width: float,
        height: float,
        obstacles: list[Polygon],
        resolution: float = 1.0,
    ) -> None:
        self.width: float = width
        self.height: float = height
        self.obstacles: list[Polygon] = obstacles
        self.resolution: float = resolution
        self._grid: npt.NDArray[np.bool_] | None = None

    # ------------------------------------------------------------------
    # Grid properties
    # ------------------------------------------------------------------

    @property
    def grid_rows(self) -> int:
        return int(np.ceil(self.height / self.resolution))

    @property
    def grid_cols(self) -> int:
        return int(np.ceil(self.width / self.resolution))

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def create_grid(self) -> npt.NDArray[np.bool_]:
        """Build (or return cached) occupancy grid.

        ``True`` means the cell is free; ``False`` means blocked.
        """
        if self._grid is not None:
            return self._grid

        rows: int = self.grid_rows
        cols: int = self.grid_cols
        grid: npt.NDArray[np.bool_] = np.ones((rows, cols), dtype=np.bool_)

        for r in range(rows):
            for c in range(cols):
                x_min: float = c * self.resolution
                y_min: float = r * self.resolution
                x_max: float = x_min + self.resolution
                y_max: float = y_min + self.resolution
                cell_box: Polygon = box(x_min, y_min, x_max, y_max)
                for obs in self.obstacles:
                    if cell_box.intersects(obs):
                        grid[r, c] = False
                        break

        self._grid = grid
        return self._grid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_free(self, row: int, col: int) -> bool:
        """Return whether the given grid cell is free (not blocked)."""
        grid: npt.NDArray[np.bool_] = self.create_grid()
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return bool(grid[row, col])
        return False

    def cell_center(self, row: int, col: int) -> Point2D:
        """Return the continuous (x, y) centre of a grid cell."""
        x: float = (col + 0.5) * self.resolution
        y: float = (row + 0.5) * self.resolution
        return Point2D(x, y)

    def xy_to_cell(self, point: Point2D) -> GridCell:
        """Convert continuous coordinates to the enclosing grid cell."""
        col: int = int(point.x / self.resolution)
        row: int = int(point.y / self.resolution)
        # Clamp to valid range
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))
        return (row, col)


# ----------------------------------------------------------------------
# Sample map factory
# ----------------------------------------------------------------------


def create_sample_map() -> MapEnvironment:
    """Return a 30x30 map with several polygon obstacles."""
    obstacles: list[Polygon] = [
        # Large L-shaped obstacle (upper-left area)
        Polygon([(4, 20), (4, 26), (6, 26), (6, 22), (10, 22), (10, 20)]),
        # Rectangle (centre)
        Polygon([(12, 12), (12, 18), (16, 18), (16, 12)]),
        # Triangle (lower-right)
        Polygon([(22, 4), (26, 4), (24, 8)]),
        # Narrow vertical wall
        Polygon([(8, 8), (8, 16), (9, 16), (9, 8)]),
        # Small square (upper-right)
        Polygon([(22, 22), (22, 26), (26, 26), (26, 22)]),
        # Irregular quadrilateral (bottom-centre)
        Polygon([(14, 2), (18, 2), (17, 6), (13, 5)]),
    ]
    return MapEnvironment(width=30.0, height=30.0, obstacles=obstacles, resolution=1.0)
