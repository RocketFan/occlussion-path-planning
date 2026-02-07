"""Plotting helpers for the occlusion-aware path planning pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from map_environment import MapEnvironment, Point2D
from visibility import compute_visibility_polygon

if TYPE_CHECKING:
    from vehicle import Vehicle


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


# ------------------------------------------------------------------
# Animation
# ------------------------------------------------------------------


def animate_vehicle(
    fig: Figure,
    ax: Axes,
    map_env: MapEnvironment,
    vehicle: "Vehicle",
    max_vis_radius: float = 15.0,
    prediction_horizon: float = 3.0,
    prediction_samples: int = 8,
    dt: float = 0.1,
    interval_ms: int = 50,
) -> FuncAnimation:
    from vehicle import Vehicle  # deferred to avoid circular import
    # Draw the static background once.
    plot_map(ax, map_env)
    plot_grid(ax, map_env)
    plot_path(ax, vehicle.path)
    ax.set_title("Vehicle Animation")

    # Mutable artists that get updated each frame.
    vehicle_dot, = ax.plot(
        [], [], marker="o", color="darkorange", markersize=10, zorder=6,
    )
    heading_line, = ax.plot(
        [], [], color="darkorange", linewidth=2, zorder=6,
    )
    # Prediction: a line + scatter dots for sampled future positions.
    prediction_line, = ax.plot(
        [], [], color="lime", linewidth=1.5, linestyle="--", zorder=5,
    )
    prediction_dots, = ax.plot(
        [], [], marker="o", linestyle="None", color="lime",
        markersize=5, zorder=5,
    )
    # Visibility patch placeholder (replaced each frame).
    vis_patches: list[MplPolygon] = []

    total_frames: int = int(np.ceil(vehicle.total_time / dt)) + 1
    vehicle.reset()

    def _init():
        vehicle_dot.set_data([], [])
        heading_line.set_data([], [])
        prediction_line.set_data([], [])
        prediction_dots.set_data([], [])
        return (vehicle_dot, heading_line, prediction_line, prediction_dots)

    def _update(frame: int):
        nonlocal vis_patches

        # Remove previous visibility patches.
        for p in vis_patches:
            p.remove()
        vis_patches.clear()

        t: float = frame * dt
        pos: Point2D = vehicle.position_at(t)
        heading: float = vehicle.heading_at(t)

        # Vehicle dot
        vehicle_dot.set_data([pos.x], [pos.y])

        # Short heading indicator line
        arrow_len: float = 1.2
        hx: float = pos.x + arrow_len * np.cos(heading)
        hy: float = pos.y + arrow_len * np.sin(heading)
        heading_line.set_data([pos.x, hx], [pos.y, hy])

        # Predicted future positions
        predictions: list[(Point2D, float)] = vehicle.predict_positions(
            t, prediction_horizon, prediction_samples,
        )
        predictions: list[Point2D] = [p[0] for p in predictions]

        if predictions:
            pred_xs: list[float] = [pos.x] + [p.x for p in predictions]
            pred_ys: list[float] = [pos.y] + [p.y for p in predictions]
            prediction_line.set_data(pred_xs, pred_ys)
            # Dots only on the sampled points (not the current position).
            prediction_dots.set_data(
                [p.x for p in predictions],
                [p.y for p in predictions],
            )
        else:
            prediction_line.set_data([], [])
            prediction_dots.set_data([], [])

        # Visibility polygon
        vis_poly: BaseGeometry = compute_visibility_polygon(
            pos, map_env.obstacles, max_vis_radius,
        )
        if not vis_poly.is_empty:
            polys: list[Polygon] = []
            if isinstance(vis_poly, MultiPolygon):
                polys = list(vis_poly.geoms)
            elif isinstance(vis_poly, Polygon):
                polys = [vis_poly]

            for poly in polys:
                coords: list[tuple[float, float]] = list(poly.exterior.coords)
                patch: MplPolygon = MplPolygon(
                    coords,
                    closed=True,
                    facecolor="yellow",
                    edgecolor="orange",
                    alpha=0.30,
                    linewidth=0.6,
                    zorder=2,
                )
                ax.add_patch(patch)
                vis_patches.append(patch)

        return (
            vehicle_dot, heading_line,
            prediction_line, prediction_dots,
            *vis_patches,
        )

    anim: FuncAnimation = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return anim
