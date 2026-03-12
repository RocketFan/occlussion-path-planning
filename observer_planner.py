"""Observer vehicle path planning with three MPC-style strategies.

Provides three planning strategies for a Dubins observer vehicle
(constant speed, bounded turn rate) to maintain line-of-sight (LOS)
with a moving target vehicle while avoiding occlusion by obstacles.

The observer state is ``(x, y, theta)`` and the control input is a
scalar turn rate ``omega in [-omega_max, omega_max]``.

Strategies:
    1. GreedyLookaheadPlanner  -- sample-based greedy forward simulation
    2. MPPIPlanner             -- Model Predictive Path Integral control
    3. ScipyMPCPlanner         -- scipy-optimised receding-horizon MPC
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
import time

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
        (
            t := _ray_segment_intersection(
                p1.x,
                p1.y,
                dx,
                dy,
                seg_a.x,
                seg_a.y,
                seg_b.x,
                seg_b.y,
            )
        )
        is None
        or not (0.0 < t < 1.0)
        for seg_a, seg_b in segments
    )


def _dubins_step(
    x: float,
    y: float,
    theta: float,
    speed: float,
    omega: float,
    dt: float,
) -> tuple[float, float, float]:
    """Advance one Dubins step using exact circular-arc integration.

    Parameters
    ----------
    x, y, theta:
        Current pose (position and heading in radians).
    speed:
        Constant forward speed.
    omega:
        Turn rate (rad/s).  Positive = counter-clockwise.
    dt:
        Time step duration.

    Returns
    -------
    (x_new, y_new, theta_new)
    """
    theta_new: float = theta + omega * dt

    # if abs(omega) > 1e-9:
    #     # Arc motion
    #     r: float = speed / omega
    #     x_new: float = x + r * (math.sin(theta_new) - math.sin(theta))
    #     y_new: float = y + r * (-math.cos(theta_new) + math.cos(theta))
    # else:
    # Straight-line motion
    x_new = x + speed * math.cos(theta_new) * dt
    y_new = y + speed * math.sin(theta_new) * dt

    return x_new, y_new, theta_new


def _dubins_steer_towards(
    x: float,
    y: float,
    theta: float,
    target_x: float,
    target_y: float,
    speed: float,
    dt: float,
    omega_max: float,
) -> tuple[float, float, float]:
    """Pick the turn rate that best steers toward a target point.

    Computes the desired heading to the target, finds the angular
    difference, converts it to a required ``omega``, clamps to
    ``[-omega_max, omega_max]``, and returns the resulting pose after
    one Dubins step.

    Returns
    -------
    (x_new, y_new, theta_new)
    """
    desired: float = math.atan2(target_y - y, target_x - x)
    diff: float = desired - theta
    # Normalise to [-pi, pi]
    diff = math.atan2(math.sin(diff), math.cos(diff))
    omega: float = diff / dt if dt > 0.0 else 0.0
    omega = max(-omega_max, min(omega_max, omega))
    return _dubins_step(x, y, theta, speed, omega, dt)


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
            target_vehicle.position_at(k * dt),
            obstacles,
            max_vis_radius,
        )
        for k in range(total_frames)
    ]
    return vis_polys


def _vis_frame(t_plan: float, sim_dt: float, total_frames: int) -> int:
    """Convert a planning time to the nearest simulation frame index."""
    return min(max(int(round(t_plan / sim_dt)), 0), total_frames - 1)


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
        current_heading: float,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        speed: float,
        dt: float,
    ) -> tuple[Point2D, float]:
        """Return the next observer position and heading."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1: Greedy with Lookahead
# ---------------------------------------------------------------------------


class GreedyLookaheadPlanner(ObserverStrategy):
    """Sample candidate turn rates, score via sequential forward simulation.

    Parameters
    ----------
    num_candidates:
        Number of evenly-spaced turn rates sampled from
        ``[-omega_max, omega_max]`` at each planning step.
    horizon_steps:
        Number of future time steps to simulate when scoring a candidate.
    preferred_distance:
        Desired distance from the target.  The planner adds a small bonus
        for positions near this distance (secondary to LOS).
    min_turning_radius:
        Minimum turning radius for the Dubins vehicle.  Together with
        ``speed`` this determines ``omega_max = speed / min_turning_radius``.
    """

    def __init__(
        self,
        num_candidates: int = 36,
        horizon_steps: int = 10,
        preferred_distance: float = 5.0,
        min_turning_radius: float = 2.5,
        planner_dt: float = 0.15,
    ) -> None:
        self._num_candidates: int = num_candidates
        self._horizon_steps: int = horizon_steps
        self._preferred_distance: float = preferred_distance
        self._min_turning_radius: float = min_turning_radius
        self._planner_dt: float = planner_dt

    def step(
        self,
        current_pos: Point2D,
        current_heading: float,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        speed: float,
        dt: float,
    ) -> tuple[Point2D, float]:
        total_frames: int = len(vis_polys)
        pref_dist: float = self._preferred_distance
        omega_max: float = speed / self._min_turning_radius
        pdt: float = self._planner_dt

        # Generate candidate turn rates: omega=0 (straight) + uniform samples
        candidate_omegas: list[float] = [0.0]
        for i in range(self._num_candidates):
            omega: float = -omega_max + 2.0 * omega_max * i / self._num_candidates
            candidate_omegas.append(omega)

        best_score: float = -float("inf")
        best_pos: Point2D = current_pos
        best_heading: float = current_heading

        for omega_c in candidate_omegas:
            # Simulate first step with this candidate turn rate (sim dt)
            sx, sy, stheta = _dubins_step(
                current_pos.x,
                current_pos.y,
                current_heading,
                speed,
                omega_c,
                dt,
            )
            sim_x, sim_y, sim_theta = sx, sy, stheta
            score: float = 0.0

            for k in range(1, self._horizon_steps + 1):
                t_plan: float = t + k * pdt
                vf: int = _vis_frame(t_plan, dt, total_frames)

                target_k: Point2D = target_vehicle.position_at(t_plan)
                vis_poly_k: Polygon = vis_polys[vf]

                if vis_poly_k.contains(Point(sim_x, sim_y)):
                    score += 1.0
                    sim_x, sim_y, sim_theta = _dubins_step(
                        sim_x,
                        sim_y,
                        sim_theta,
                        speed,
                        0.0,
                        pdt,
                    )
                else:
                    sim_x, sim_y, sim_theta = _dubins_steer_towards(
                        sim_x,
                        sim_y,
                        sim_theta,
                        target_k.x,
                        target_k.y,
                        speed,
                        pdt,
                        omega_max,
                    )

                # Small bonus for being near the preferred distance (secondary
                # to the integer LOS score which dominates).
                dist_to_tgt: float = math.hypot(
                    sim_x - target_k.x,
                    sim_y - target_k.y,
                )
                dist_err: float = abs(dist_to_tgt - pref_dist)
                score += 0.05 / (1.0 + dist_err)

            # Bonus: prefer being deep inside the next-step visibility polygon
            candidate_frame: int = frame + 1
            if candidate_frame < total_frames:
                sd: float = _signed_distance_to_vis_boundary(
                    Point2D(sx, sy),
                    vis_polys[candidate_frame],
                )
                score += sd * 0.01

            if score > best_score:
                best_score = score
                best_pos = Point2D(sx, sy)
                best_heading = stheta

        return best_pos, best_heading


# ---------------------------------------------------------------------------
# Strategy 2: MPPI
# ---------------------------------------------------------------------------


class MPPIPlanner(ObserverStrategy):
    """Model Predictive Path Integral trajectory optimisation (Dubins).

    The control input is a sequence of scalar turn rates ``omega``.

    Parameters
    ----------
    K:
        Number of sampled trajectory perturbations.
    horizon_steps:
        Planning horizon length (number of future time steps).
    sigma:
        Standard deviation of the Gaussian noise added to the nominal
        control sequence.
    lambda_:
        Temperature parameter for the exponential weighting of
        trajectory scores.
    preferred_distance:
        Desired distance from the target.  The distance reward peaks
        when the observer is at this distance and falls off for
        deviations in either direction (secondary to LOS).
    min_turning_radius:
        Minimum turning radius for the Dubins vehicle.
    """

    def __init__(
        self,
        K: int = 128,
        horizon_steps: int = 10,
        sigma: float = 1.5,
        lambda_: float = 0.1,
        preferred_distance: float = 5.0,
        min_turning_radius: float = 2.5,
        planner_dt: float = 0.15,
    ) -> None:
        self._K: int = K
        self._horizon_steps: int = horizon_steps
        self._sigma: float = sigma
        self._lambda: float = lambda_
        self._preferred_distance: float = preferred_distance
        self._min_turning_radius: float = min_turning_radius
        self._planner_dt: float = planner_dt
        self._nominal_U: np.ndarray | None = None

    def step(
        self,
        current_pos: Point2D,
        current_heading: float,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        speed: float,
        dt: float,
    ) -> tuple[Point2D, float]:
        H: int = self._horizon_steps
        K: int = self._K
        total_frames: int = len(vis_polys)
        pref_dist: float = self._preferred_distance
        omega_max: float = speed / self._min_turning_radius
        pdt: float = self._planner_dt
        w_los: float = 10.0
        w_dist: float = 1.0

        # Initialise / warm-start nominal control sequence (H,) -- turn rates
        if self._nominal_U is None or self._nominal_U.shape[0] != H:
            self._nominal_U = np.zeros(H)

        U: np.ndarray = self._nominal_U.copy()

        # Sample K perturbations (1-D noise for each horizon step)
        noise: np.ndarray = np.random.randn(K, H) * self._sigma
        U_samples: np.ndarray = U[np.newaxis, :] + noise  # (K, H)

        # Clamp each turn rate to [-omega_max, omega_max]
        U_samples = np.clip(U_samples, -omega_max, omega_max)

        # Roll out each trajectory and score
        scores: np.ndarray = np.zeros(K)

        for k_idx in range(K):
            px: float = current_pos.x
            py: float = current_pos.y
            ptheta: float = current_heading

            for i in range(H):
                px, py, ptheta = _dubins_step(
                    px,
                    py,
                    ptheta,
                    speed,
                    float(U_samples[k_idx, i]),
                    pdt,
                )
                t_i: float = t + (i + 1) * pdt
                vf: int = _vis_frame(t_i, dt, total_frames)
                if vf >= total_frames:
                    break

                target_i: Point2D = target_vehicle.position_at(t_i)

                obs_pos: Point2D = Point2D(px, py)
                los: bool = has_line_of_sight(obs_pos, target_i, segments)

                dist: float = math.hypot(px - target_i.x, py - target_i.y)
                dist_err: float = abs(dist - pref_dist)
                scores[k_idx] += w_los * float(los) + w_dist / (1.0 + dist_err)

        # Compute weights (with numerical stability)
        scores -= np.max(scores)
        weights: np.ndarray = np.exp(scores / self._lambda)
        weights /= np.sum(weights)

        # Update nominal sequence: weighted average
        U_new: np.ndarray = np.einsum("k,ki->i", weights, U_samples)
        self._nominal_U = U_new

        # Apply first action with simulation dt
        omega0: float = float(np.clip(self._nominal_U[0], -omega_max, omega_max))
        nx, ny, ntheta = _dubins_step(
            current_pos.x,
            current_pos.y,
            current_heading,
            speed,
            omega0,
            dt,
        )

        # Shift for warm start
        self._nominal_U = np.append(self._nominal_U[1:], 0.0)

        return Point2D(nx, ny), ntheta


# ---------------------------------------------------------------------------
# Strategy 3: Scipy Optimisation MPC
# ---------------------------------------------------------------------------
class ScipyMPCPlanner(ObserverStrategy):
    """Receding-horizon MPC solved via ``scipy.optimize.minimize`` (Dubins).

    The decision variable is a sequence of scalar turn rates
    ``[omega_1, ..., omega_H]``.  The solver used is ``L-BFGS-B`` with
    box bounds ``[-omega_max, omega_max]`` on each turn rate (no
    constraint function needed).

    Parameters
    ----------
    horizon_steps:
        Planning horizon length.
    w_los:
        Weight for the LOS penalty (penalises being outside the
        visibility polygon).  This is intentionally the largest weight
        so that maintaining visibility dominates the cost.
    w_dist:
        Weight for the distance-preference penalty.  Penalises deviation
        from *preferred_distance* rather than raw distance to target.
    w_smooth:
        Weight for the smoothness / turn-rate jerk penalty.
    preferred_distance:
        Desired distance from the target.  The cost term is
        ``w_dist * (dist - preferred_distance)^2``.
    min_turning_radius:
        Minimum turning radius for the Dubins vehicle.
    """

    def __init__(
        self,
        horizon_steps: int = 10,
        w_los: float = 10.0,
        w_dist: float = 1.0,
        w_smooth: float = 0.5,
        preferred_distance: float = 5.0,
        min_turning_radius: float = 2.5,
        planner_dt: float = 0.15,
    ) -> None:
        self._horizon_steps: int = horizon_steps
        self._w_los: float = w_los
        self._w_dist: float = w_dist
        self._w_smooth: float = w_smooth
        self._preferred_distance: float = preferred_distance
        self._min_turning_radius: float = min_turning_radius
        self._planner_dt: float = planner_dt
        self._prev_omegas: np.ndarray | None = None

    def step(
        self,
        current_pos: Point2D,
        current_heading: float,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        speed: float,
        dt: float,
    ) -> tuple[Point2D, float]:
        from scipy.optimize import minimize

        H: int = self._horizon_steps
        total_frames: int = len(vis_polys)
        omega_max: float = speed / self._min_turning_radius
        pdt: float = self._planner_dt

        # Pre-compute target positions and vis polys for the horizon
        target_positions: list[Point2D] = []
        horizon_vis_polys: list[Polygon | None] = []
        for k in range(1, H + 1):
            t_k: float = t + k * pdt
            vf: int = _vis_frame(t_k, dt, total_frames)
            target_positions.append(target_vehicle.position_at(t_k))
            if vf < total_frames:
                horizon_vis_polys.append(vis_polys[vf])
            else:
                horizon_vis_polys.append(None)

        # Warm-start initial guess (turn-rate sequence)
        if self._prev_omegas is not None and len(self._prev_omegas) == H:
            x0: np.ndarray = np.zeros(H)
            # Shift: drop old first, append zero
            x0[: H - 1] = self._prev_omegas[1:]
            x0[H - 1] = 0.0
        else:
            x0 = np.zeros(H)

        # Capture in local scope for closures
        w_los: float = self._w_los
        w_dist: float = self._w_dist
        w_smooth: float = self._w_smooth
        pref_dist: float = self._preferred_distance
        cx: float = current_pos.x
        cy: float = current_pos.y
        ctheta: float = current_heading
        spd: float = speed

        def cost(omega_seq: np.ndarray) -> float:
            J: float = 0.0
            px, py, ptheta = cx, cy, ctheta

            for k in range(H):
                omega_k: float = float(omega_seq[k])
                px, py, ptheta = _dubins_step(px, py, ptheta, spd, omega_k, pdt)

                vis_poly: Polygon | None = horizon_vis_polys[k]
                if vis_poly is not None:
                    sd: float = _signed_distance_to_vis_boundary(
                        Point2D(px, py),
                        vis_poly,
                    )
                    if sd < 0:
                        J += w_los * sd * sd

                tgt: Point2D = target_positions[k]
                dist_to_tgt: float = math.hypot(px - tgt.x, py - tgt.y)
                dist_err: float = dist_to_tgt - pref_dist
                J += w_dist * dist_err * dist_err

            for k in range(1, H):
                d_omega: float = float(omega_seq[k] - omega_seq[k - 1])
                J += w_smooth * d_omega * d_omega

            return J

        # Box bounds on each turn rate
        bounds = [(-omega_max, omega_max)] * H

        result = minimize(cost, x0, method="SLSQP", bounds=bounds)
        self._prev_omegas = result.x.copy()

        # Apply first turn rate with simulation dt
        omega0: float = float(np.clip(result.x[0], -omega_max, omega_max))
        nx, ny, ntheta = _dubins_step(
            current_pos.x,
            current_pos.y,
            current_heading,
            speed,
            omega0,
            dt,
        )

        return Point2D(nx, ny), ntheta


# ---------------------------------------------------------------------------
# Strategy 4: Waypoint MPC (Dubins-constrained)
# ---------------------------------------------------------------------------


class WaypointMPCPlanner(ObserverStrategy):
    """Receding-horizon MPC over N waypoint positions, Dubins-constrained.

    Instead of optimising a sequence of turn rates, the decision
    variables are ``N`` waypoint *positions* ``(x, y)`` that the
    observer should fly over.  The cost is evaluated directly at the
    waypoint positions (no forward rollout needed).  Feasibility is
    enforced by requiring that consecutive waypoints are reachable via
    a Dubins path within the allocated time budget.

    After optimisation the planner computes the shortest Dubins path
    from the current pose to the first waypoint and returns the
    configuration one ``speed * dt`` step along that path.

    Parameters
    ----------
    num_waypoints:
        Number of waypoint positions to optimise.
    horizon_steps:
        Planning horizon length (in time steps).  The waypoints are
        evenly distributed across this horizon.
    w_los:
        Weight for the LOS penalty (visibility polygon).
    w_dist:
        Weight for the distance-preference term.
    preferred_distance:
        Desired distance from the target.
    min_turning_radius:
        Minimum turning radius for the Dubins vehicle.
    """

    def __init__(
        self,
        num_waypoints: int = 2,
        horizon_steps: int = 20,
        w_los: float = 10.0,
        w_dist: float = 1.0,
        preferred_distance: float = 5.0,
        min_turning_radius: float = 2.5,
        planner_dt: float = 0.15,
    ) -> None:
        self._num_waypoints: int = num_waypoints
        self._horizon_steps: int = horizon_steps
        self._w_los: float = w_los
        self._w_dist: float = w_dist
        self._preferred_distance: float = preferred_distance
        self._min_turning_radius: float = min_turning_radius
        self._planner_dt: float = planner_dt
        self._prev_waypoints: np.ndarray | None = None

    def step(
        self,
        current_pos: Point2D,
        current_heading: float,
        t: float,
        frame: int,
        target_vehicle: Vehicle,
        segments: list[tuple[Point2D, Point2D]],
        vis_polys: list[Polygon],
        speed: float,
        dt: float,
    ) -> tuple[Point2D, float]:
        from scipy.optimize import minimize

        from dubins_path import (
            dubins_path_length,
            dubins_path_sample,
            dubins_shortest_path,
        )

        N: int = self._num_waypoints
        H: int = self._horizon_steps
        total_frames: int = len(vis_polys)
        rho: float = self._min_turning_radius
        pdt: float = self._planner_dt

        # Pre-compute target positions and vis polys for each waypoint
        target_positions: list[Point2D] = []
        horizon_vis_polys: list[Polygon | None] = []
        for i in range(N):
            t_k: float = t + (i + 1) * N / H * pdt
            vf: int = _vis_frame(t_k, dt, total_frames)
            target_positions.append(target_vehicle.position_at(t_k))
            if vf < total_frames:
                horizon_vis_polys.append(vis_polys[vf])
            else:
                horizon_vis_polys.append(None)

        # Time budget per segment (evenly divided)
        seg_time: float = (H / N) * pdt
        seg_dist_budget: float = speed * seg_time

        # ---- Warm-start / initial guess --------------------------------
        if self._prev_waypoints is not None and len(self._prev_waypoints) == 2 * N:
            x0: np.ndarray = np.empty(2 * N)
            # Shift: previous wp2..wpN become new wp1..wp_{N-1}
            x0[: 2 * (N - 1)] = self._prev_waypoints[2:]
            # New last waypoint starts at previous last waypoint position
            x0[2 * (N - 1)] = self._prev_waypoints[-2]
            x0[2 * (N - 1) + 1] = self._prev_waypoints[-1]
        else:
            x0 = np.empty(2 * N)
            for i in range(N):
                tgt = target_positions[i]
                angle_to_tgt = math.atan2(
                    tgt.y - current_pos.y, tgt.x - current_pos.x
                )
                x0[2 * i] = tgt.x - self._preferred_distance * math.cos(angle_to_tgt)
                x0[2 * i + 1] = tgt.y - self._preferred_distance * math.sin(angle_to_tgt)

        # Capture locals for closures
        w_los: float = self._w_los
        w_dist: float = self._w_dist
        pref_dist: float = self._preferred_distance
        cx: float = current_pos.x
        cy: float = current_pos.y
        ctheta: float = current_heading

        def _derive_headings(
            flat_vars: np.ndarray,
        ) -> list[float]:
            """Compute heading at each waypoint from the waypoint chain."""
            wps = flat_vars.reshape(N, 2)
            headings: list[float] = []
            for i in range(N):
                if i < N - 1:
                    headings.append(
                        math.atan2(wps[i + 1, 1] - wps[i, 1], wps[i + 1, 0] - wps[i, 0])
                    )
                else:
                    tgt = target_positions[i]
                    headings.append(
                        math.atan2(tgt.y - wps[i, 1], tgt.x - wps[i, 0])
                    )
            return headings

        def cost(flat_vars: np.ndarray) -> float:
            wps = flat_vars.reshape(N, 2)
            J: float = 0.0
            for i in range(N):
                wx, wy = float(wps[i, 0]), float(wps[i, 1])

                vis_poly: Polygon | None = horizon_vis_polys[i]
                if vis_poly is not None:
                    sd: float = _signed_distance_to_vis_boundary(
                        Point2D(wx, wy), vis_poly
                    )
                    if sd < 0:
                        J += w_los * sd * sd

                tgt: Point2D = target_positions[i]
                dist_to_tgt: float = math.hypot(wx - tgt.x, wy - tgt.y)
                dist_err: float = dist_to_tgt - pref_dist
                J += w_dist * dist_err * dist_err
            return J

        def dubins_constraint(flat_vars: np.ndarray) -> np.ndarray:
            """Return array of (budget - path_length) per segment; >= 0 is feasible."""
            wps = flat_vars.reshape(N, 2)
            headings = _derive_headings(flat_vars)
            residuals = np.empty(N)

            prev_x, prev_y, prev_theta = cx, cy, ctheta
            for i in range(N):
                wx, wy = float(wps[i, 0]), float(wps[i, 1])
                wtheta = headings[i]
                path = dubins_shortest_path(
                    (prev_x, prev_y, prev_theta),
                    (wx, wy, wtheta),
                    rho,
                )
                if path is None:
                    residuals[i] = -1e6
                else:
                    residuals[i] = seg_dist_budget - dubins_path_length(path)
                prev_x, prev_y, prev_theta = wx, wy, wtheta
            return residuals

        constraints = {"type": "ineq", "fun": dubins_constraint}

        result = minimize(
            cost,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )
        self._prev_waypoints = result.x.copy()

        # ---- Execute: Dubins path to first waypoint -------------------
        opt_wps = result.x.reshape(N, 2)
        headings = _derive_headings(result.x)
        wp1_x, wp1_y = float(opt_wps[0, 0]), float(opt_wps[0, 1])
        wp1_theta = headings[0]

        path = dubins_shortest_path(
            (cx, cy, ctheta), (wp1_x, wp1_y, wp1_theta), rho
        )
        if path is not None:
            step_dist: float = speed * dt
            nx, ny, ntheta = dubins_path_sample(path, step_dist)
        else:
            nx, ny, ntheta = _dubins_step(cx, cy, ctheta, speed, 0.0, dt)

        return Point2D(nx, ny), ntheta


# ---------------------------------------------------------------------------
# Main planning loop
# ---------------------------------------------------------------------------


def plan_observer_path(
    strategy: ObserverStrategy,
    target_vehicle: Vehicle,
    obstacles: list[Polygon],
    start_pos: Point2D,
    start_heading: float,
    speed: float,
    dt: float,
    max_vis_radius: float,
) -> tuple[list[Point2D], list[float]]:
    """Plan the observer path using the given strategy.

    Returns ``(positions, headings)`` -- one entry per ``dt`` time step.
    """
    segments: list[tuple[Point2D, Point2D]] = _collect_segments(obstacles)
    vis_polys: list[Polygon] = precompute_target_visibility(
        target_vehicle,
        obstacles,
        dt,
        max_vis_radius,
    )
    total_frames: int = len(vis_polys)

    positions: list[Point2D] = [start_pos]
    headings: list[float] = [start_heading]
    current_pos: Point2D = start_pos
    current_heading: float = start_heading

    for frame in range(total_frames - 1):
        t: float = frame * dt
        next_pos, next_heading = strategy.step(
            current_pos,
            current_heading,
            t,
            frame,
            target_vehicle,
            segments,
            vis_polys,
            speed,
            dt,
        )
        positions.append(next_pos)
        headings.append(next_heading)
        current_pos = next_pos
        current_heading = next_heading

    return positions, headings
