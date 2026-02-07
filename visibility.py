"""2-D visibility polygon computation via ray casting."""

from __future__ import annotations

import math
from typing import Sequence

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from map_environment import Point2D

# Small angular offset used when casting rays near obstacle vertices.
_EPSILON: float = 1e-5


def _collect_segments(obstacles: list[Polygon]) -> list[tuple[Point2D, Point2D]]:
    """Extract every edge segment from a list of Shapely polygons."""
    segments: list[tuple[Point2D, Point2D]] = []
    for obs in obstacles:
        coords: Sequence[tuple[float, float]] = list(obs.exterior.coords)
        for i in range(len(coords) - 1):
            a: Point2D = Point2D(coords[i][0], coords[i][1])
            b: Point2D = Point2D(coords[i + 1][0], coords[i + 1][1])
            segments.append((a, b))
    return segments


def _ray_segment_intersection(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float | None:
    """Return the ray parameter *t* for the closest intersection, or None.

    The ray starts at (ox, oy) with direction (dx, dy).
    The segment runs from (ax, ay) to (bx, by).
    """
    denom: float = dx * (by - ay) - dy * (bx - ax)
    if abs(denom) < 1e-12:
        return None  # parallel

    t: float = ((ax - ox) * (by - ay) - (ay - oy) * (bx - ax)) / denom
    u: float = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def compute_visibility_polygon(
    observer: Point2D,
    obstacles: list[Polygon],
    max_radius: float,
) -> Polygon:
    """Compute the 2-D visibility polygon from *observer*.

    Parameters
    ----------
    observer:
        The viewpoint in continuous coordinates.
    obstacles:
        List of Shapely polygons acting as opaque blockers.
    max_radius:
        Maximum sight distance; the visibility region is clipped to a
        circle of this radius around the observer.

    Returns
    -------
    Polygon
        A Shapely polygon representing the visible area.
    """
    segments: list[tuple[Point2D, Point2D]] = _collect_segments(obstacles)

    # Collect unique angles toward every obstacle vertex.
    angles: list[float] = []
    for obs in obstacles:
        for vx, vy in obs.exterior.coords:
            angle: float = math.atan2(vy - observer.y, vx - observer.x)
            # Cast three rays per vertex for robust corner handling.
            angles.extend([angle - _EPSILON, angle, angle + _EPSILON])

    # Also cast rays toward the bounding circle perimeter to fill gaps.
    num_circle_rays: int = 360
    for i in range(num_circle_rays):
        angles.append(2.0 * math.pi * i / num_circle_rays)

    # De-duplicate (close-enough) and sort
    angles.sort()

    # Cast each ray and find closest hit.
    hit_points: list[Point2D] = []
    ox: float = observer.x
    oy: float = observer.y

    for angle in angles:
        dx: float = math.cos(angle)
        dy: float = math.sin(angle)

        closest_t: float = max_radius  # default: hit the bounding circle

        for seg_a, seg_b in segments:
            t: float | None = _ray_segment_intersection(
                ox, oy, dx, dy, seg_a.x, seg_a.y, seg_b.x, seg_b.y
            )
            if t is not None and t < closest_t:
                closest_t = t

        hx: float = ox + dx * closest_t
        hy: float = oy + dy * closest_t
        hit_points.append(Point2D(hx, hy))

    if len(hit_points) < 3:
        # Degenerate case: return a tiny circle around the observer.
        return Point(ox, oy).buffer(0.01)

    # Build the polygon from the sorted hit points.
    vis_poly: Polygon | MultiPolygon = Polygon(hit_points)
    if not vis_poly.is_valid:
        vis_poly = vis_poly.buffer(0)  # attempt to fix self-intersections

    # Clip to bounding circle.
    bounding_circle: Polygon = Point(ox, oy).buffer(max_radius, resolution=64)
    clipped = vis_poly.intersection(bounding_circle)

    # The intersection / buffer can produce a MultiPolygon.
    # Collapse it to the single largest Polygon for downstream consumers.
    if isinstance(clipped, MultiPolygon):
        largest: Polygon = max(clipped.geoms, key=lambda g: g.area)
        return largest

    if isinstance(clipped, Polygon):
        return clipped

    # Fallback for any other geometry type (e.g. GeometryCollection).
    return Point(ox, oy).buffer(0.01)


def compute_path_visibility(
    path: list[Point2D],
    obstacles: list[Polygon],
    max_radius: float,
) -> list[Polygon]:
    """Compute the visibility polygon at every point along *path*.

    Returns a list with the same length as *path*.
    """
    visibility_polygons: list[Polygon] = [
        compute_visibility_polygon(pt, obstacles, max_radius) for pt in path
    ]
    return visibility_polygons
