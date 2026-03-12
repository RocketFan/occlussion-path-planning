"""Shortest Dubins path solver.

Computes the shortest feasible path between two configurations
``(x, y, theta)`` for a vehicle with a minimum turning radius,
considering all six Dubins path types (LSL, LSR, RSL, RSR, LRL, RLR).

Public API
----------
- ``dubins_shortest_path(q0, q1, rho)`` -- compute the shortest path.
- ``dubins_path_length(path)``          -- total arc-length of a path.
- ``dubins_path_sample(path, t)``       -- sample ``(x, y, theta)`` at
                                           arc-length *t* along the path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple


TWO_PI: float = 2.0 * math.pi


def _mod2pi(angle: float) -> float:
    """Normalise *angle* into ``[0, 2*pi)``."""
    return angle % TWO_PI


class SegmentKind(Enum):
    LEFT = auto()
    STRAIGHT = auto()
    RIGHT = auto()


class _DubinsParams(NamedTuple):
    """Normalised parameters for a candidate Dubins path."""
    t: float
    p: float
    q: float


@dataclass(frozen=True, slots=True)
class DubinsPath:
    """A fully resolved Dubins path.

    Attributes
    ----------
    qi : tuple[float, float, float]
        Start configuration ``(x, y, theta)``.
    segments : tuple[float, float, float]
        Normalised segment lengths (multiply by *rho* for metric length).
    segment_kinds : tuple[SegmentKind, SegmentKind, SegmentKind]
        Arc type for each of the three segments.
    rho : float
        Minimum turning radius used.
    """
    qi: tuple[float, float, float]
    segments: tuple[float, float, float]
    segment_kinds: tuple[SegmentKind, SegmentKind, SegmentKind]
    rho: float

    @property
    def total_length(self) -> float:
        return (self.segments[0] + self.segments[1] + self.segments[2]) * self.rho


# ------------------------------------------------------------------
# Candidate path formulas (normalised coordinates, radius = 1)
# ------------------------------------------------------------------

def _lsl(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p_sq < 0.0:
        return None
    p = math.sqrt(p_sq)
    t = _mod2pi(-alpha + math.atan2(cb - ca, d + sa - sb))
    q = _mod2pi(beta - math.atan2(cb - ca, d + sa - sb))
    return _DubinsParams(t, p, q)


def _rsr(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sb - sa)
    if p_sq < 0.0:
        return None
    p = math.sqrt(p_sq)
    t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb))
    q = _mod2pi(-beta + math.atan2(ca - cb, d - sa + sb))
    return _DubinsParams(t, p, q)


def _lsr(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_sq = -2.0 + d * d + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return None
    p = math.sqrt(p_sq)
    t = _mod2pi(-alpha + math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p))
    q = _mod2pi(-beta + math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p))
    return _DubinsParams(t, p, q)


def _rsl(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_sq = -2.0 + d * d + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return None
    p = math.sqrt(p_sq)
    t = _mod2pi(alpha - math.atan2(ca + cb, d - sa - sb) + math.atan2(2.0, p))
    q = _mod2pi(beta - math.atan2(ca + cb, d - sa - sb) + math.atan2(2.0, p))
    return _DubinsParams(t, p, q)


def _rlr(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    val = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(val) > 1.0:
        return None
    p = _mod2pi(TWO_PI - math.acos(val))
    t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return _DubinsParams(t, p, q)


def _lrl(alpha: float, beta: float, d: float) -> _DubinsParams | None:
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    val = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sb - sa)) / 8.0
    if abs(val) > 1.0:
        return None
    p = _mod2pi(TWO_PI - math.acos(val))
    t = _mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + _mod2pi(p / 2.0))
    q = _mod2pi(_mod2pi(beta) - alpha + t - _mod2pi(p))
    return _DubinsParams(t, p, q)


_PATH_TYPES: list[
    tuple[
        type[_DubinsParams] | None,
        tuple[SegmentKind, SegmentKind, SegmentKind],
    ]
] = [
    (_lsl, (SegmentKind.LEFT, SegmentKind.STRAIGHT, SegmentKind.LEFT)),
    (_lsr, (SegmentKind.LEFT, SegmentKind.STRAIGHT, SegmentKind.RIGHT)),
    (_rsl, (SegmentKind.RIGHT, SegmentKind.STRAIGHT, SegmentKind.LEFT)),
    (_rsr, (SegmentKind.RIGHT, SegmentKind.STRAIGHT, SegmentKind.RIGHT)),
    (_rlr, (SegmentKind.RIGHT, SegmentKind.LEFT, SegmentKind.RIGHT)),
    (_lrl, (SegmentKind.LEFT, SegmentKind.RIGHT, SegmentKind.LEFT)),
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def dubins_shortest_path(
    q0: tuple[float, float, float],
    q1: tuple[float, float, float],
    rho: float,
) -> DubinsPath | None:
    """Compute the shortest Dubins path from *q0* to *q1*.

    Parameters
    ----------
    q0 : (x, y, theta)
        Start configuration.
    q1 : (x, y, theta)
        End configuration.
    rho : float
        Minimum turning radius (must be > 0).

    Returns
    -------
    DubinsPath or None
        The shortest feasible path, or ``None`` if no path exists
        (should not happen for valid inputs).
    """
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    d = math.hypot(dx, dy) / rho
    theta = math.atan2(dy, dx)
    alpha = _mod2pi(q0[2] - theta)
    beta = _mod2pi(q1[2] - theta)

    best: DubinsPath | None = None
    best_cost: float = float("inf")

    for solver, kinds in _PATH_TYPES:
        params = solver(alpha, beta, d)
        if params is None:
            continue
        cost = params.t + params.p + params.q
        if cost < best_cost:
            best_cost = cost
            best = DubinsPath(
                qi=q0,
                segments=(params.t, params.p, params.q),
                segment_kinds=kinds,
                rho=rho,
            )

    return best


def dubins_path_length(path: DubinsPath) -> float:
    """Return the total arc-length of a Dubins path."""
    return path.total_length


def dubins_path_sample(
    path: DubinsPath,
    t: float,
) -> tuple[float, float, float]:
    """Sample the configuration ``(x, y, theta)`` at arc-length *t*.

    Parameters
    ----------
    path :
        A ``DubinsPath`` returned by :func:`dubins_shortest_path`.
    t : float
        Distance along the path (0 = start, total_length = end).
        Clamped to ``[0, total_length]``.
    """
    t = max(0.0, min(t, path.total_length))
    t_norm = t / path.rho

    qi = path.qi
    x, y, theta = qi[0], qi[1], qi[2]

    for seg_len, kind in zip(path.segments, path.segment_kinds):
        if t_norm < seg_len:
            x, y, theta = _step_segment(x, y, theta, t_norm, kind, path.rho)
            return x, y, theta
        x, y, theta = _step_segment(x, y, theta, seg_len, kind, path.rho)
        t_norm -= seg_len

    return x, y, theta


def _step_segment(
    x: float,
    y: float,
    theta: float,
    seg_len: float,
    kind: SegmentKind,
    rho: float,
) -> tuple[float, float, float]:
    """Advance along one segment of a Dubins path."""
    if kind == SegmentKind.LEFT:
        x += rho * (math.sin(theta + seg_len) - math.sin(theta))
        y += rho * (-math.cos(theta + seg_len) + math.cos(theta))
        theta += seg_len
    elif kind == SegmentKind.RIGHT:
        x += rho * (-math.sin(theta - seg_len) + math.sin(theta))
        y += rho * (math.cos(theta - seg_len) - math.cos(theta))
        theta -= seg_len
    else:  # STRAIGHT
        x += rho * seg_len * math.cos(theta)
        y += rho * seg_len * math.sin(theta)
    return x, y, theta
