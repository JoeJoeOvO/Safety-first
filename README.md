# [T-ASE'25] CBF-Based Hierarchical Quadratic Programs with Guaranteed Feasibility for Safety-Critical Systems

This repository reproduces the algorithm called **Safety-first CLF-CBF QP** in our paper 

"[CBF-Based Hierarchical Quadratic Programs with Guaranteed Feasibility for Safety-Critical Systems](https://ieeexplore.ieee.org/document/11230627)", *IEEE Transactions on Automation Science and Engineering*, 2025.

[![Video](https://img.shields.io/badge/dynamic/json?style=flat&label=%E2%96%B7%20Video&query=%24.metrics%5B%22safetyfirst-video%22%5D.display&url=https%3A%2F%2Fraw.githubusercontent.com%2FJoeJoeOvO%2FJoeJoeOVO.github.io%2Fmaster%2Fdata%2Fresource-metrics.json&labelColor=fff9f7&color=fff0ec)](https://www.bilibili.com/video/BV1yk2MBUE13)

The code demonstrates the pipeline used in the paper:

1. Input polygonal obstacle boundary points.
2. Approximate each obstacle by multiple circles using constrained Delaunay triangulation (CDT).
3. Build one CBF per approximating circle.
4. Compute safe controls for a unicycle model with CLF-CBF-QP variants.

## 🧮 Algorithms

The public example keeps the following methods:

- `ClfCbfQp`: standard CLF-CBF QP
- `OptimalDecay`: Optimal-decay CLF-CBF QP (with optimal decay/slack variables)
- `SafetyFirst`: the proposed safety-first hierarchical CLF-CBF QP


## ⚙️ Installation

```bash
git clone https://github.com/JoeJoeOvO/Safety-first.git
cd Safety-first
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Run Examples

Run the narrow-corridor example:

```bash
python examples/run_simulation.py --scene narrow_corridor
```

Run the multi-obstacle example:

```bash
python examples/run_simulation.py --scene multi_obstacle
```

Run only the proposed method:

```bash
python examples/run_simulation.py --scene narrow_corridor --methods SafetyFirst
```

Generated figures are written to `outputs/`.

## 📁 Repository Layout

```text
safetyfirst/
  controllers.py      CLF-CBF QP, Optimal-decay CLF-CBF QP, and our **SafetyFirst CLF-CBF QP** controllers
  geometry.py         CDT-based circle approximation from obstacle boundaries
  scenes.py           Two example maps: narrow corridor and multi-obstacle
  simulation.py       Unicycle simulation and plotting utilities
examples/
  run_simulation.py   Command-line example runner
requirements.txt      Python dependencies
```

## 📖 Citation

If our research is useful for you, please cite:

```bibtex
@ARTICLE{11230627,
  author={Xie, Junjun and Hu, Liang and Tan, Yunzhe and Yang, Jun},
  journal={IEEE Transactions on Automation Science and Engineering},
  title={CBF-Based Hierarchical Quadratic Programs With Guaranteed Feasibility for Safety-Critical Systems},
  year={2025},
  volume={22},
  pages={23687-23699},
  doi={10.1109/TASE.2025.3629713}
}
```
