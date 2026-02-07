"""Vehicle that follows a planned path on the grid."""

from __future__ import annotations

import math

from map_environment import Point2D


class Vehicle:
    """A point vehicle that moves along a pre-planned path at a given speed.

    The vehicle linearly interpolates between consecutive waypoints.
    """

    def __init__(self, path: list[Point2D], speed: float = 2.0) -> None:
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        self._path: list[Point2D] = path
        self._speed: float = speed

        # Pre-compute cumulative arc-length distances along the path.
        self._cumulative_dist: list[float] = [0.0]
        for i in range(1, len(path)):
            dx: float = path[i].x - path[i - 1].x
            dy: float = path[i].y - path[i - 1].y
            seg_len: float = math.hypot(dx, dy)
            self._cumulative_dist.append(self._cumulative_dist[-1] + seg_len)

        self._total_length: float = self._cumulative_dist[-1]
        self._elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> list[Point2D]:
        return self._path

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def total_length(self) -> float:
        """Total arc-length of the path."""
        return self._total_length

    @property
    def total_time(self) -> float:
        """Time needed to traverse the entire path at the current speed."""
        return self._total_length / self._speed

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def finished(self) -> bool:
        return self._elapsed >= self.total_time

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset elapsed time to zero."""
        self._elapsed = 0.0

    def step(self, dt: float) -> Point2D:
        """Advance the vehicle by *dt* seconds and return its new position."""
        self._elapsed = min(self._elapsed + dt, self.total_time)
        return self.position_at(self._elapsed)

    def position_at(self, t: float) -> Point2D:
        """Return the interpolated position at time *t* (seconds)."""
        dist: float = min(t * self._speed, self._total_length)

        # Binary-search for the segment containing *dist*.
        lo: int = 0
        hi: int = len(self._cumulative_dist) - 1
        while lo < hi - 1:
            mid: int = (lo + hi) // 2
            if self._cumulative_dist[mid] <= dist:
                lo = mid
            else:
                hi = mid

        seg_start_dist: float = self._cumulative_dist[lo]
        seg_end_dist: float = self._cumulative_dist[hi]
        seg_length: float = seg_end_dist - seg_start_dist

        if seg_length < 1e-12:
            return self._path[lo]

        alpha: float = (dist - seg_start_dist) / seg_length
        alpha = max(0.0, min(1.0, alpha))

        x: float = self._path[lo].x + alpha * (self._path[hi].x - self._path[lo].x)
        y: float = self._path[lo].y + alpha * (self._path[hi].y - self._path[lo].y)
        return Point2D(x, y)

    def heading_at(self, t: float) -> float:
        """Return the heading angle (radians) at time *t*."""
        dist: float = min(t * self._speed, self._total_length)

        lo: int = 0
        hi: int = len(self._cumulative_dist) - 1
        while lo < hi - 1:
            mid: int = (lo + hi) // 2
            if self._cumulative_dist[mid] <= dist:
                lo = mid
            else:
                hi = mid

        dx: float = self._path[hi].x - self._path[lo].x
        dy: float = self._path[hi].y - self._path[lo].y
        return math.atan2(dy, dx)
