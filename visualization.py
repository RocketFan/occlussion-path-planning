"""Plotting helpers for the occlusion-aware path planning pipeline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from map_environment import MapEnvironment, Point2D


def plot_map(ax: Axes, map_env: MapEnvironment) -> None:
    """Draw the map boundary and filled obstacle polygons."""
    # Map boundary
    ax.set_xlim(0, map_env.width)
    ax.set_ylim(0, map_env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Map with Obstacles")

    for obs in map_env.obstacles:
        coords: list[tuple[float, float]] = list(obs.exterior.coords)
        patch: MplPolygon = MplPolygon(
            coords, closed=True, facecolor="dimgray", edgecolor="black", linewidth=1.2
        )
        ax.add_patch(patch)


def plot_grid(ax: Axes, map_env: MapEnvironment) -> None:
    """Overlay the occupancy grid (free cells light, blocked cells dark)."""
    grid: npt.NDArray[np.bool_] = map_env.create_grid()

    # Display as an image: row 0 is at the bottom (origin='lower').
    ax.imshow(
        ~grid,  # invert so blocked=white on a dark colour-map
        extent=[0, map_env.width, 0, map_env.height],
        origin="lower",
        cmap="Greys",
        alpha=0.35,
        interpolation="nearest",
    )
    ax.set_xlim(0, map_env.width)
    ax.set_ylim(0, map_env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Occupancy Grid Overlay")


def plot_path(ax: Axes, path: list[Point2D]) -> None:
    """Draw the A* path with start / goal markers."""
    xs: list[float] = [p.x for p in path]
    ys: list[float] = [p.y for p in path]

    ax.plot(xs, ys, color="royalblue", linewidth=2, zorder=3, label="A* path")
    ax.plot(xs[0], ys[0], marker="o", color="green", markersize=10, zorder=4, label="Start")
    ax.plot(xs[-1], ys[-1], marker="*", color="red", markersize=14, zorder=4, label="Goal")
    ax.legend(loc="upper right", fontsize=8)


def _draw_polygon(ax: Axes, polygon: Polygon) -> None:
    """Draw a single Shapely Polygon as a matplotlib patch."""
    coords: list[tuple[float, float]] = list(polygon.exterior.coords)
    patch: MplPolygon = MplPolygon(
        coords,
        closed=True,
        facecolor="yellow",
        edgecolor="orange",
        alpha=0.35,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(patch)


def plot_visibility(
    ax: Axes,
    observer: Point2D,
    visibility_polygon: BaseGeometry,
) -> None:
    """Draw a single visibility polygon with the observer point.

    Handles both ``Polygon`` and ``MultiPolygon`` geometries.
    """
    if visibility_polygon.is_empty:
        return

    if isinstance(visibility_polygon, MultiPolygon):
        for poly in visibility_polygon.geoms:
            _draw_polygon(ax, poly)
    elif isinstance(visibility_polygon, Polygon):
        _draw_polygon(ax, visibility_polygon)

    ax.plot(
        observer.x,
        observer.y,
        marker="D",
        color="darkorange",
        markersize=6,
        zorder=5,
    )


def plot_all_visibility(
    ax: Axes,
    path: list[Point2D],
    visibility_polygons: list[Polygon],
) -> None:
    """Draw visibility polygons for every waypoint along the path."""
    for observer, vis_poly in zip(path, visibility_polygons):
        plot_visibility(ax, observer, vis_poly)
    ax.set_title("Visibility along Path")
