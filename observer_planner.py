"""Observer vehicle path planning with three MPC-style strategies.

Provides three planning strategies for an observer vehicle to maintain
line-of-sight (LOS) with a moving target vehicle while respecting speed
constraints and avoiding occlusion by obstacles.

Strategies:
    1. GreedyLookaheadPlanner  -- sample-based greedy forward simulation
    2. MPPIPlanner             -- Model Predictive Path Integral control
    3. ScipyMPCPlanner         -- scipy-optimised receding-horizon MPC
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from shapely.geometry import Point, Polygon

from map_environment import Point2D
from vehicle import Vehicle
from visibility import (
    _collect_segments,
    _ray_segment_intersection,
    compute_visibility_polygon,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def has_line_of_sight(
    p1: Point2D,
    p2: Point2D,
    segments: list[tuple[Point2D, Point2D]],
) -> bool:
    """Return True if there is a clear line of sight from *p1* to *p2*.

    Casts a ray from *p1* toward *p2*; if any obstacle segment is hit
    with ``0 < t < 1``, LOS is blocked.
    """
    dx: float = p2.x - p1.x
    dy: float = p2.y - p1.y

    return all(
        (t := _ray_segment_intersection(
            p1.x, p1.y, dx, dy,
            seg_a.x, seg_a.y, seg_b.x, seg_b.y,
        )) is None or not (0.0 < t < 1.0)
        for seg_a, seg_b in segments
    )


def _move_towards(current: Point2D, target: Point2D, max_move: float) -> Point2D:
    """Move from *current* toward *target* by at most *max_move* distance."""
    dx: float = target.x - current.x
    dy: float = target.y - current.y
    dist: float = math.hypot(dx, dy)

    if dist <= max_move or dist < 1e-12:
        return target

    ratio: float = max_move / dist
    return Point2D(current.x + dx * ratio, current.y + dy * ratio)


def precompute_target_visibility(
    target_vehicle: Vehicle,
    obstacles: list[Polygon],
    dt: float,
    max_vis_radius: float,
) -> list[Polygon]:
    """Pre-compute the visibility polygon ``V(t_k)`` for each time step.

    Returns a list indexed by frame number: ``vis_polys[k]`` is the
    visibility polygon when the target is at
    ``target_vehicle.position_at(k * dt)``.
    """
    total_frames: int = int(math.ceil(target_vehicle.total_time / dt)) + 1
    vis_polys: list[Polygon] = [
        compute_visibility_polygon(
            target_vehicle.position_at(k * dt), obstacles, max_vis_radius,
        )
        for k in range(total_frames)
    ]
    return vis_polys


def _signed_distance_to_vis_boundary(point: Point2D, vis_poly: Polygon) -> float:
    """Signed distance to the visibility polygon boundary.

    Positive if the point is **inside** (has LOS), negative if **outside**.
    """
    p: Point = Point(point.x, point.y)
    dist: float = vis_poly.exterior.distance(p)
    if vis_poly.contains(p):
        return dist
    return -dist


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class ObserverStrategy(ABC):
    """Base class for observer planning strategies."""

    @abstractmethod
    def step(
        self,
        current_pos: Point2D,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        max_speed: float,
        dt: float,
    ) -> Point2D:
        """Return the next observer position."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1: Greedy with Lookahead
# ---------------------------------------------------------------------------


class GreedyLookaheadPlanner(ObserverStrategy):
    """Sample candidate positions, score via sequential forward simulation."""

    def __init__(
        self,
        num_candidates: int = 36,
        horizon_steps: int = 10,
    ) -> None:
        self._num_candidates: int = num_candidates
        self._horizon_steps: int = horizon_steps

    def step(
        self,
        current_pos: Point2D,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        max_speed: float,
        dt: float,
    ) -> Point2D:
        max_move: float = max_speed * dt
        total_frames: int = len(vis_polys)

        # Generate candidates: stay + points on the reachable circle
        candidates: list[Point2D] = [current_pos]
        for i in range(self._num_candidates):
            angle: float = 2.0 * math.pi * i / self._num_candidates
            cx: float = current_pos.x + max_move * math.cos(angle)
            cy: float = current_pos.y + max_move * math.sin(angle)
            candidates.append(Point2D(cx, cy))

        best_score: float = -float("inf")
        best_candidate: Point2D = current_pos

        for c in candidates:
            sim_pos: Point2D = c
            score: float = 0.0

            for k in range(1, self._horizon_steps + 1):
                future_frame: int = frame + k
                if future_frame >= total_frames:
                    break

                target_k: Point2D = target_vehicle.position_at((frame + k) * dt)
                vis_poly_k: Polygon = vis_polys[future_frame]

                if vis_poly_k.contains(Point(sim_pos.x, sim_pos.y)):
                    score += 1.0
                    # LOS maintained -- stay at sim_pos
                else:
                    # No LOS -- greedily move toward target
                    sim_pos = _move_towards(sim_pos, target_k, max_move)

            # Bonus: prefer being deep inside the next-step visibility polygon
            candidate_frame: int = frame + 1
            if candidate_frame < total_frames:
                sd: float = _signed_distance_to_vis_boundary(
                    c, vis_polys[candidate_frame],
                )
                score += sd * 0.01

            if score > best_score:
                best_score = score
                best_candidate = c

        return best_candidate


# ---------------------------------------------------------------------------
# Strategy 2: MPPI
# ---------------------------------------------------------------------------


class MPPIPlanner(ObserverStrategy):
    """Model Predictive Path Integral trajectory optimisation."""

    def __init__(
        self,
        K: int = 128,
        horizon_steps: int = 10,
        sigma: float = 1.5,
        lambda_: float = 0.1,
    ) -> None:
        self._K: int = K
        self._horizon_steps: int = horizon_steps
        self._sigma: float = sigma
        self._lambda: float = lambda_
        self._nominal_U: np.ndarray | None = None

    def step(
        self,
        current_pos: Point2D,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        max_speed: float,
        dt: float,
    ) -> Point2D:
        H: int = self._horizon_steps
        K: int = self._K
        total_frames: int = len(vis_polys)
        w_los: float = 10.0
        w_dist: float = 1.0

        # Initialise / warm-start nominal control sequence
        if self._nominal_U is None or self._nominal_U.shape[0] != H:
            self._nominal_U = np.zeros((H, 2))

        U: np.ndarray = self._nominal_U.copy()

        # Sample K perturbations
        noise: np.ndarray = np.random.randn(K, H, 2) * self._sigma
        U_samples: np.ndarray = U[np.newaxis, :, :] + noise  # (K, H, 2)

        # Clamp each velocity to max_speed
        norms: np.ndarray = np.linalg.norm(U_samples, axis=2, keepdims=True)
        mask: np.ndarray = norms > max_speed
        U_samples = np.where(mask, U_samples * (max_speed / np.maximum(norms, 1e-12)), U_samples)

        # Roll out each trajectory and score
        scores: np.ndarray = np.zeros(K)

        for k_idx in range(K):
            pos: np.ndarray = np.array([current_pos.x, current_pos.y])

            for i in range(H):
                pos = pos + U_samples[k_idx, i] * dt
                future_frame: int = frame + i + 1
                if future_frame >= total_frames:
                    break

                t_i: float = t + (i + 1) * dt
                target_i: Point2D = target_vehicle.position_at(t_i)

                obs_pos: Point2D = Point2D(pos[0], pos[1])
                los: bool = has_line_of_sight(obs_pos, target_i, segments)

                dist: float = math.hypot(pos[0] - target_i.x, pos[1] - target_i.y)
                scores[k_idx] += w_los * float(los) + w_dist / (1.0 + dist)

        # Compute weights (with numerical stability)
        scores -= np.max(scores)
        weights: np.ndarray = np.exp(scores / self._lambda)
        weights /= np.sum(weights)

        # Update nominal sequence: weighted average
        U_new: np.ndarray = np.einsum("k,kij->ij", weights, U_samples)
        self._nominal_U = U_new

        # Apply first action (clamped to max_speed)
        u0: np.ndarray = self._nominal_U[0].copy()
        speed: float = np.linalg.norm(u0)
        if speed > max_speed:
            u0 *= max_speed / speed

        next_pos: Point2D = Point2D(
            current_pos.x + u0[0] * dt,
            current_pos.y + u0[1] * dt,
        )

        # Shift for warm start
        self._nominal_U = np.vstack([self._nominal_U[1:], np.zeros((1, 2))])

        return next_pos


# ---------------------------------------------------------------------------
# Strategy 3: Scipy Optimisation MPC
# ---------------------------------------------------------------------------


class ScipyMPCPlanner(ObserverStrategy):
    """Receding-horizon MPC solved via ``scipy.optimize.minimize``."""

    def __init__(
        self,
        horizon_steps: int = 10,
        w_los: float = 10.0,
        w_dist: float = 1.0,
        w_smooth: float = 0.5,
        w_speed: float = 100.0,
    ) -> None:
        self._horizon_steps: int = horizon_steps
        self._w_los: float = w_los
        self._w_dist: float = w_dist
        self._w_smooth: float = w_smooth
        self._w_speed: float = w_speed
        self._prev_trajectory: np.ndarray | None = None

    def step(
        self,
        current_pos: Point2D,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        max_speed: float,
        dt: float,
    ) -> Point2D:
        from scipy.optimize import minimize

        H: int = self._horizon_steps
        total_frames: int = len(vis_polys)
        max_move: float = max_speed * dt

        # Pre-compute target positions and vis polys for the horizon
        target_positions: list[Point2D] = []
        horizon_vis_polys: list[Polygon | None] = []
        for k in range(1, H + 1):
            future_frame: int = frame + k
            t_k: float = t + k * dt
            target_positions.append(target_vehicle.position_at(t_k))
            if future_frame < total_frames:
                horizon_vis_polys.append(vis_polys[future_frame])
            else:
                horizon_vis_polys.append(None)

        # Warm-start initial guess
        if self._prev_trajectory is not None and len(self._prev_trajectory) == H * 2:
            x0: np.ndarray = np.zeros(H * 2)
            # Shift: drop old first position, append copy of last
            x0[: 2 * (H - 1)] = self._prev_trajectory[2:]
            x0[2 * (H - 1) :] = self._prev_trajectory[-2:]
        else:
            x0 = np.tile([current_pos.x, current_pos.y], H)

        # Capture weights in local scope for the closure
        w_los: float = self._w_los
        w_dist: float = self._w_dist
        w_smooth: float = self._w_smooth
        w_speed: float = self._w_speed
        cx: float = current_pos.x
        cy: float = current_pos.y

        def cost(traj_flat: np.ndarray) -> float:
            J: float = 0.0

            # Build full position array: pos[0] = current, pos[1..H] = decision
            all_x: np.ndarray = np.empty(H + 1)
            all_y: np.ndarray = np.empty(H + 1)
            all_x[0] = cx
            all_y[0] = cy
            for k in range(H):
                all_x[k + 1] = traj_flat[2 * k]
                all_y[k + 1] = traj_flat[2 * k + 1]

            for k in range(H):
                xk: float = all_x[k + 1]
                yk: float = all_y[k + 1]

                # --- LOS term (penalise being outside visibility polygon)
                vis_poly: Polygon | None = horizon_vis_polys[k]
                if vis_poly is not None:
                    sd: float = _signed_distance_to_vis_boundary(
                        Point2D(xk, yk), vis_poly,
                    )
                    if sd < 0:
                        J += w_los * sd * sd

                # --- Distance term
                tgt: Point2D = target_positions[k]
                J += w_dist * ((xk - tgt.x) ** 2 + (yk - tgt.y) ** 2)

                # --- Speed-limit penalty
                dx: float = xk - all_x[k]
                dy: float = yk - all_y[k]
                step_dist: float = math.hypot(dx, dy)
                excess: float = step_dist - max_move
                if excess > 0:
                    J += w_speed * excess * excess

            # --- Smoothness (acceleration penalty)
            for k in range(1, H):
                ax: float = all_x[k + 1] - 2.0 * all_x[k] + all_x[k - 1]
                ay: float = all_y[k + 1] - 2.0 * all_y[k] + all_y[k - 1]
                J += w_smooth * (ax * ax + ay * ay)

            return J

        result = minimize(cost, x0, method="L-BFGS-B")
        self._prev_trajectory = result.x.copy()

        # Apply first position, clamped to max speed
        next_x: float = result.x[0]
        next_y: float = result.x[1]

        dx: float = next_x - current_pos.x
        dy: float = next_y - current_pos.y
        dist: float = math.hypot(dx, dy)
        if dist > max_move:
            ratio: float = max_move / dist
            next_x = current_pos.x + dx * ratio
            next_y = current_pos.y + dy * ratio

        return Point2D(next_x, next_y)


# ---------------------------------------------------------------------------
# Main planning loop
# ---------------------------------------------------------------------------


def plan_observer_path(
    strategy: ObserverStrategy,
    target_vehicle: Vehicle,
    obstacles: list[Polygon],
    start_pos: Point2D,
    max_speed: float,
    dt: float,
    max_vis_radius: float,
) -> list[Point2D]:
    """Plan the observer path using the given strategy.

    Returns a list of ``Point2D`` positions, one per ``dt`` time step.
    """
    segments: list[tuple[Point2D, Point2D]] = _collect_segments(obstacles)
    vis_polys: list[Polygon] = precompute_target_visibility(
        target_vehicle, obstacles, dt, max_vis_radius,
    )
    total_frames: int = len(vis_polys)

    positions: list[Point2D] = [start_pos]
    current_pos: Point2D = start_pos

    for frame in range(total_frames - 1):
        t: float = frame * dt
        next_pos: Point2D = strategy.step(
            current_pos, t, frame,
            target_vehicle, segments, vis_polys,
            max_speed, dt,
        )
        positions.append(next_pos)
        current_pos = next_pos

    return positions
