from __future__ import annotations

from typing import Iterable

import numpy as np
import triangle


Circle = tuple[float, float, float]
Point = tuple[float, float]


def _circumcircle(points: np.ndarray) -> Circle | None:
    a, b, c = points
    d = 2.0 * (
        a[0] * (b[1] - c[1])
        + b[0] * (c[1] - a[1])
        + c[0] * (a[1] - b[1])
    )
    if abs(d) < 1e-12:
        return None
    center_x = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    center_y = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    radius = float(np.linalg.norm(np.array([center_x, center_y]) - a))
    if not np.isfinite(radius) or radius <= 1e-9:
        return None
    return float(center_x), float(center_y), radius


def _densify_boundary(points: np.ndarray, max_edge_length: float) -> np.ndarray:
    if max_edge_length <= 0:
        raise ValueError("max_edge_length must be positive.")

    dense_points: list[np.ndarray] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        edge_length = float(np.linalg.norm(end - start))
        segment_count = max(1, int(np.ceil(edge_length / max_edge_length)))
        for segment_index in range(segment_count):
            ratio = segment_index / segment_count
            dense_points.append(start + ratio * (end - start))
    return np.array(dense_points, dtype=float)


def circles_from_boundary_points(
    boundary_points: Iterable[Iterable[float]],
    max_edge_length: float = 0.35,
) -> tuple[list[Circle], np.ndarray]:
    """Approximate one polygonal obstacle with CDT circumcircles."""
    polygon = np.array(list(boundary_points), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("boundary_points must contain at least three 2D points.")

    polygon = _densify_boundary(polygon, max_edge_length)
    segments = np.array([[i, (i + 1) % len(polygon)] for i in range(len(polygon))])
    triangulation = triangle.triangulate({"vertices": polygon, "segments": segments}, "p")

    circles: list[Circle] = []
    for triangle_indices in triangulation.get("triangles", []):
        circle = _circumcircle(triangulation["vertices"][triangle_indices])
        if circle is not None:
            circles.append(circle)
    closed_polygon = np.vstack([polygon, polygon[0]])
    return circles, closed_polygon


def circles_from_polygons(
    polygons: Iterable[Iterable[Iterable[float]]],
    max_edge_length: float = 0.35,
) -> tuple[list[Circle], list[np.ndarray]]:
    circles: list[Circle] = []
    refined_polygons: list[np.ndarray] = []
    for polygon in polygons:
        obstacle_circles, refined_polygon = circles_from_boundary_points(polygon, max_edge_length)
        circles.extend(obstacle_circles)
        refined_polygons.append(refined_polygon)
    return circles, refined_polygons
