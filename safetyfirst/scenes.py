from __future__ import annotations

from dataclasses import dataclass

from .geometry import Point


@dataclass(frozen=True)
class Scene:
    name: str
    start: tuple[float, float, float]
    goal: Point
    polygons: tuple[tuple[Point, ...], ...]
    max_edge_length: float = 0.35
    sensor_range: float | None = 0.9
    xlim: tuple[float, float] = (-2.0, 2.0)
    ylim: tuple[float, float] = (-2.0, 3.0)


NARROW_CORRIDOR = Scene(
    name="narrow_corridor",
    start=(0.0, -1.65, 1.57079632679),
    goal=(0.0, 2.45),
    polygons=(
        ((-1.65, -2.65), (-1.15, -2.65), (-1.15, 3.20), (-1.65, 3.20)),
        ((1.15, -2.65), (1.65, -2.65), (1.65, 3.20), (1.15, 3.20)),
        ((0.24, -0.95), (1.15, -0.95), (1.15, -0.42), (0.24, -0.42)),
        ((-1.15, 0.12), (-0.02, 0.12), (-0.02, 0.65), (-1.15, 0.65)),
        ((0.12, 1.18), (1.15, 1.18), (1.15, 1.70), (0.12, 1.70)),
    ),
    max_edge_length=0.42,
    sensor_range=0.75,
    xlim=(-1.9, 1.9),
    ylim=(-2.8, 3.3),
)


MULTI_OBSTACLE = Scene(
    name="multi_obstacle",
    start=(-1.65, -0.35, 0.18),
    goal=(1.65, 0.22),
    polygons=(
        ((-1.22, 0.18), (-0.92, 0.04), (-0.70, 0.26), (-0.84, 0.62), (-1.20, 0.58)),
        ((-1.22, -0.88), (-0.82, -0.78), (-0.66, -0.48), (-0.98, -0.28), (-1.30, -0.50)),
        ((-0.46, 0.54), (-0.10, 0.36), (0.16, 0.62), (-0.02, 0.94), (-0.42, 0.88)),
        ((-0.36, -0.64), (0.10, -0.82), (0.36, -0.50), (0.12, -0.20), (-0.28, -0.26)),
        ((0.32, 0.04), (0.82, -0.10), (1.14, 0.22), (0.90, 0.60), (0.38, 0.48)),
        ((0.58, -1.06), (1.04, -0.94), (1.20, -0.62), (0.84, -0.40), (0.50, -0.66)),
        ((1.10, 0.62), (1.42, 0.48), (1.62, 0.72), (1.42, 0.98), (1.08, 0.88)),
    ),
    max_edge_length=0.20,
    sensor_range=0.50,
    xlim=(-1.9, 1.9),
    ylim=(-1.1, 1.1),
)


SCENES = {
    NARROW_CORRIDOR.name: NARROW_CORRIDOR,
    MULTI_OBSTACLE.name: MULTI_OBSTACLE,
}


def get_scene(name: str) -> Scene:
    try:
        return SCENES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(SCENES))
        raise ValueError(f"Unknown scene '{name}'. Supported scenes: {supported}") from exc
