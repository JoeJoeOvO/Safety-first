from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetyfirst import SCENES, get_scene, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Safety-first CLF-CBF-QP examples.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENES),
        default="narrow_corridor",
        help="Example map to simulate.",
    )
    parser.add_argument(
        "--methods",
        default="ClfCbfQp,OptimalDecay,SafetyFirst",
        help="Comma-separated methods: ClfCbfQp, OptimalDecay, SafetyFirst.",
    )
    parser.add_argument("--sim-time", type=float, default=18.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    scene = get_scene(args.scene)
    output_dir = Path(args.output_dir) / scene.name
    results = run_simulation(
        scene,
        methods=methods,
        sim_time=args.sim_time,
        time_step=args.dt,
        output_dir=output_dir,
    )
    for result in results:
        print(
            f"{result.method:12s} status={result.status:10s} "
            f"steps={result.steps:4d} goal_error={result.final_error:.3f} "
            f"min_cbf={result.min_cbf:.3f} mean_qp={result.mean_solve_time_ms:.2f} ms"
        )
        if result.figure_path is not None:
            print(f"  figure: {result.figure_path}")


if __name__ == "__main__":
    main()
