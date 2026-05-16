from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

from .controllers import ControllerParams, SafetyFirstController
from .geometry import circles_from_polygons
from .scenes import Scene


METHODS = ("ClfCbfQp", "OptimalDecay", "SafetyFirst")


@dataclass
class SimulationResult:
    method: str
    status: str
    steps: int
    final_state: tuple[float, float, float]
    final_error: float
    min_cbf: float
    mean_solve_time_ms: float
    figure_path: Path | None


def _system(_, state: list[float], control: np.ndarray) -> np.ndarray:
    v, omega = control
    return np.array([v * math.cos(state[2]), v * math.sin(state[2]), omega])


def _minimum_geometric_clearance(
    state: list[float],
    circles: list[tuple[float, float, float]],
) -> float:
    x, y, _ = state
    return min(math.hypot(x - cx, y - cy) - radius for cx, cy, radius in circles)


def _plot_result(
    scene: Scene,
    refined_polygons: list[np.ndarray],
    circles: list[tuple[float, float, float]],
    states: list[list[float]],
    method: str,
    status: str,
    output_dir: Path | None,
) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{scene.name}_{method}.png"

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.set_aspect("equal", adjustable="box")
    for polygon in refined_polygons:
        ax.fill(polygon[:, 0], polygon[:, 1], color="#9CA3AF", alpha=0.35)
        ax.plot(polygon[:, 0], polygon[:, 1], color="#374151", linewidth=0.8)
    for cx, cy, radius in circles:
        ax.add_patch(Circle((cx, cy), radius, color="#6B7280", alpha=0.08))
    trajectory = np.array(states)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#DC2626", linewidth=2.0, label=method)
    if status == "collision":
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], marker="x", markersize=11, markeredgewidth=3, color="#111827", label="Collision")
    ax.plot(scene.start[0], scene.start[1], marker="s", color="#2563EB", label="Start")
    ax.plot(scene.goal[0], scene.goal[1], marker="*", markersize=12, color="#16A34A", label="Goal")
    ax.set_xlim(*scene.xlim)
    ax.set_ylim(*scene.ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title(f"{scene.name} - {method}")
    fig.savefig(figure_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return figure_path


def run_simulation(
    scene: Scene,
    methods: tuple[str, ...] = METHODS,
    sim_time: float = 18.0,
    time_step: float = 0.02,
    output_dir: str | Path | None = "outputs",
    params: ControllerParams | None = None,
    collision_depth: float = 0.0,
) -> list[SimulationResult]:
    unsupported = sorted(set(methods) - set(METHODS))
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")

    circles, refined_polygons = circles_from_polygons(scene.polygons, scene.max_edge_length)
    resolved_output = None if output_dir is None else Path(output_dir)
    results: list[SimulationResult] = []

    for method in methods:
        controller = SafetyFirstController(circles, scene.goal, params=params, sensor_range=scene.sensor_range)
        state = list(scene.start)
        states = [state.copy()]
        min_cbf = float("inf")
        solve_times = []
        status = "timeout"
        t_now = 0.0

        while t_now <= sim_time:
            if controller.goal_distance(state) < controller.params.goal_radius:
                status = "reached"
                break
            start = __import__("time").perf_counter()
            control = controller.control(state, method)
            solve_times.append(__import__("time").perf_counter() - start)
            if control is None:
                status = "infeasible"
                break
            solution = solve_ivp(_system, (t_now, t_now + time_step), state, args=(control,), rtol=1e-5, atol=1e-7)
            state = [float(solution.y[0, -1]), float(solution.y[1, -1]), float(solution.y[2, -1])]
            states.append(state.copy())
            t_now += time_step
            values = controller.cbf_values(state)
            if values:
                min_cbf = min(min_cbf, min(values))
                if _minimum_geometric_clearance(state, circles) < -collision_depth:
                    status = "collision"
                    break

        figure_path = _plot_result(scene, refined_polygons, circles, states, method, status, resolved_output)
        mean_time = float("nan") if not solve_times else 1000.0 * sum(solve_times) / len(solve_times)
        results.append(
            SimulationResult(
                method=method,
                status=status,
                steps=len(solve_times),
                final_state=(state[0], state[1], state[2]),
                final_error=controller.goal_distance(state),
                min_cbf=min_cbf,
                mean_solve_time_ms=mean_time,
                figure_path=figure_path,
            )
        )
    return results
