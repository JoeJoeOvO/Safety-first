from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from cvxopt import matrix, solvers

from .geometry import Circle, Point


solvers.options["show_progress"] = False


@dataclass
class ControllerParams:
    lam_clf: float = 2.0
    gamma_cbf: float = 5.0
    clf_slack_weight: float = 10.0
    cbf_slack_weight: float = 20.0
    heading_clf_weight: float = 0.5
    heading_cbf_weight: float = 0.1
    v_bounds: tuple[float, float] = (0.0, 2.0)
    omega_bounds: tuple[float, float] = (-1.2, 1.2)
    min_forward_speed: float = 0.04
    speed_decay_rate: float = 0.5
    goal_radius: float = 0.08
    max_active_obstacles: int = 12


class SafetyFirstController:
    def __init__(
        self,
        circles: Iterable[Circle],
        goal: Point,
        params: ControllerParams | None = None,
        sensor_range: float | None = None,
    ) -> None:
        self.circles = list(circles)
        self.goal = goal
        self.params = params or ControllerParams()
        self.sensor_range = sensor_range
        self.v_bounds = list(self.params.v_bounds)

    def active_obstacles(self, state: Iterable[float]) -> list[int]:
        if self.sensor_range is None:
            return list(range(len(self.circles)))
        x, y, _ = state
        distances = []
        for index, (cx, cy, radius) in enumerate(self.circles):
            clearance = math.hypot(x - cx, y - cy) - radius
            if clearance < self.sensor_range:
                distances.append((clearance, index))
        distances.sort()
        return [index for _, index in distances[: self.params.max_active_obstacles]]

    def control(
        self,
        state: Iterable[float],
        method: str,
        nominal_control: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray | None:
        if method == "ClfCbfQp":
            return self._clf_cbf_qp(state, nominal_control)
        if method == "OptimalDecay":
            return self._optimal_decay_qp(state, nominal_control)
        if method == "SafetyFirst":
            return self._safety_first_qp(state, nominal_control)
        raise ValueError("method must be ClfCbfQp, OptimalDecay, or SafetyFirst.")

    def goal_distance(self, state: Iterable[float]) -> float:
        x, y, _ = state
        return max(math.hypot(x - self.goal[0], y - self.goal[1]), 1e-9)

    def clf_value(self, state: Iterable[float]) -> float:
        x, y, theta = state
        dx = x - self.goal[0]
        dy = y - self.goal[1]
        distance = self.goal_distance(state)
        projection = dx * math.cos(theta) + dy * math.sin(theta)
        return dx**2 + dy**2 + self.params.heading_clf_weight * (projection / distance + 1.0)

    def cbf_values(self, state: Iterable[float]) -> list[float]:
        return [self.cbf_value(state, index) for index in range(len(self.circles))]

    def cbf_value(self, state: Iterable[float], index: int) -> float:
        x, y, theta = state
        cx, cy, radius = self.circles[index]
        dx = x - cx
        dy = y - cy
        distance = max(math.hypot(dx, dy), 1e-9)
        projection = dx * math.cos(theta) + dy * math.sin(theta)
        return (
            math.sqrt(distance)
            - math.sqrt(radius)
            + self.params.heading_cbf_weight * (projection / distance - 1.0)
        )

    def _clf_gradient(self, state: Iterable[float]) -> tuple[float, float, float]:
        x, y, theta = state
        dx = x - self.goal[0]
        dy = y - self.goal[1]
        distance = self.goal_distance(state)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        projection = dx * cos_theta + dy * sin_theta
        distance_cubed = distance**3
        weight = self.params.heading_clf_weight
        grad_x = 2.0 * dx + weight * (cos_theta / distance - projection * dx / distance_cubed)
        grad_y = 2.0 * dy + weight * (sin_theta / distance - projection * dy / distance_cubed)
        grad_theta = weight * (-dx * sin_theta + dy * cos_theta) / distance
        return grad_x, grad_y, grad_theta

    def _cbf_gradient(self, state: Iterable[float], index: int) -> tuple[float, float, float]:
        x, y, theta = state
        cx, cy, _ = self.circles[index]
        dx = x - cx
        dy = y - cy
        distance = max(math.hypot(dx, dy), 1e-9)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        projection = dx * cos_theta + dy * sin_theta
        weight = self.params.heading_cbf_weight
        distance_cubed = distance**3
        radial_scale = 0.5 / (distance ** 1.5)
        grad_x = radial_scale * dx + weight * (cos_theta / distance - projection * dx / distance_cubed)
        grad_y = radial_scale * dy + weight * (sin_theta / distance - projection * dy / distance_cubed)
        grad_theta = weight * (-dx * sin_theta + dy * cos_theta) / distance
        return grad_x, grad_y, grad_theta

    def _clf_row(self, state: Iterable[float], include_slack: bool = True) -> list[float]:
        _, _, theta = state
        grad_x, grad_y, grad_theta = self._clf_gradient(state)
        row = [grad_x * math.cos(theta) + grad_y * math.sin(theta), grad_theta]
        if include_slack:
            row.append(-1.0)
        return [float(value) for value in row]

    def _cbf_row(self, state: Iterable[float], index: int, include_slack: bool = True) -> list[float]:
        _, _, theta = state
        grad_x, grad_y, grad_theta = self._cbf_gradient(state, index)
        row = [-(grad_x * math.cos(theta) + grad_y * math.sin(theta)), -grad_theta]
        if include_slack:
            row.append(0.0)
        return [float(value) for value in row]

    def _update_speed_lower_bound(self, state: Iterable[float], active_values: list[float]) -> None:
        min_speed = 0.0 if self.goal_distance(state) < self.params.goal_radius else self.params.min_forward_speed
        if active_values and min(active_values) >= 0.0:
            decayed_speed = (1.0 - math.exp(-self.params.speed_decay_rate * self.goal_distance(state))) * self.params.v_bounds[1]
            self.v_bounds[0] = max(min_speed, decayed_speed)
        else:
            self.v_bounds[0] = min_speed

    def _add_input_constraints(self, rows: list[list[float]], bounds: list[float], width: int) -> None:
        zeros = [0.0 for _ in range(width - 2)]
        rows.extend(
            [
                [-1.0, 0.0] + zeros,
                [1.0, 0.0] + zeros,
                [0.0, -1.0] + zeros,
                [0.0, 1.0] + zeros,
            ]
        )
        bounds.extend(
            [
                -self.v_bounds[0],
                self.v_bounds[1],
                -self.params.omega_bounds[0],
                self.params.omega_bounds[1],
            ]
        )

    @staticmethod
    def _solve_qp(H: list[list[float]], p: list[float], rows: list[list[float]], bounds: list[float]) -> np.ndarray | None:
        try:
            solution = solvers.qp(matrix(H), matrix(p), matrix(np.array(rows, dtype=float)), matrix(bounds))
        except Exception:
            return None
        values = np.array(solution["x"], dtype=float).reshape(-1)
        if solution.get("status") == "optimal":
            return values
        residual = np.array(rows, dtype=float) @ values - np.array(bounds, dtype=float)
        if np.max(residual) <= 1e-5:
            return values
        return None

    def _prepare_active(self, state: Iterable[float]) -> tuple[list[int], list[float]]:
        active = self.active_obstacles(state)
        active_values = [self.cbf_value(state, index) for index in active]
        self._update_speed_lower_bound(state, active_values)
        return active, active_values

    def _clf_cbf_qp(self, state: Iterable[float], nominal_control: tuple[float, float]) -> np.ndarray | None:
        active, active_values = self._prepare_active(state)
        H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, self.params.clf_slack_weight]]
        p = [-nominal_control[0], -nominal_control[1], 0.0]
        rows = [self._cbf_row(state, index) for index in active]
        bounds = [self.params.gamma_cbf * value for value in active_values]
        rows.append(self._clf_row(state))
        bounds.append(-self.params.lam_clf * self.clf_value(state))
        self._add_input_constraints(rows, bounds, 3)
        solution = self._solve_qp(H, p, rows, bounds)
        return None if solution is None else solution[:2]

    def _optimal_decay_qp(self, state: Iterable[float], nominal_control: tuple[float, float]) -> np.ndarray | None:
        active, active_values = self._prepare_active(state)
        width = 3 + len(active)
        H = np.zeros((width, width), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = self.params.clf_slack_weight
        for slack_index in range(len(active)):
            H[3 + slack_index, 3 + slack_index] = self.params.cbf_slack_weight
        p = [-nominal_control[0], -nominal_control[1], 0.0] + [0.0 for _ in active]

        rows: list[list[float]] = []
        bounds: list[float] = []
        for local_index, obstacle_index in enumerate(active):
            row = self._cbf_row(state, obstacle_index) + [0.0 for _ in active]
            row[3 + local_index] = 1.0
            rows.append(row)
            bounds.append(self.params.gamma_cbf * active_values[local_index])
        rows.append(self._clf_row(state) + [0.0 for _ in active])
        bounds.append(-self.params.lam_clf * self.clf_value(state))
        self._add_input_constraints(rows, bounds, width)
        solution = self._solve_qp(H.tolist(), p, rows, bounds)
        return None if solution is None else solution[:2]

    def _safety_first_qp(self, state: Iterable[float], nominal_control: tuple[float, float]) -> np.ndarray | None:
        active, active_values = self._prepare_active(state)
        ranked = [active[index] for index in np.argsort(np.array(active_values))]
        value_by_index = dict(zip(active, active_values))
        feasibility_H = [[1e-9, 0.0, 0.0], [0.0, 1e-9, 0.0], [0.0, 0.0, 1.0]]
        feasibility_p = [0.0, 0.0, 0.0]
        rows: list[list[float]] = []
        bounds: list[float] = []
        self._add_input_constraints(rows, bounds, 3)

        for obstacle_index in ranked:
            candidate_row = self._cbf_row(state, obstacle_index, include_slack=False) + [1.0]
            candidate_bound = self.params.gamma_cbf * value_by_index[obstacle_index]
            solution = self._solve_qp(feasibility_H, feasibility_p, rows + [candidate_row], bounds + [candidate_bound])
            if solution is None:
                return None
            relaxation = 0.0 if solution[2] >= 0.0 or abs(solution[2]) < 1e-5 else 1.001 * solution[2]
            rows.append(self._cbf_row(state, obstacle_index))
            bounds.append(candidate_bound - relaxation)

        clf_candidate_row = self._clf_row(state)
        clf_candidate_bound = -self.params.lam_clf * self.clf_value(state)
        solution = self._solve_qp(feasibility_H, feasibility_p, rows + [clf_candidate_row], bounds + [clf_candidate_bound])
        if solution is None:
            return None
        relaxation = 0.0 if solution[2] <= 0.0 or abs(solution[2]) < 1e-5 else 1.001 * solution[2]
        rows.append(self._clf_row(state, include_slack=False) + [0.0])
        bounds.append(clf_candidate_bound + relaxation)

        control_rows = [row[:2] for row in rows]
        H = [[1.0, 0.0], [0.0, 1.0]]
        p = [-nominal_control[0], -nominal_control[1]]
        return self._solve_qp(H, p, control_rows, bounds)
