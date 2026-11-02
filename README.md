[README.md](https://github.com/user-attachments/files/34278065/README.md)
# DATA_Greece_Aqua2

Field data and code for the paper **"Neural-based feedback control on an Autonomous Surface Vehicle"** by Ioannis Faros and Herbert G. Tanner (University of Delaware), currently under review at *Frontiers*.

The data were collected with the Aqua2 robot in Corinth, Greece, in June 2026. Aqua2 was driven from the origin to target poses $`(x_d, y_d, \theta_d)`$ by a model-based controller, about 10 times per target. The success rate for each target is used to train a neural network that approximates

```math
V(q) = -\log \mathbb{P}(\text{reach goal from } q),
```

whose gradient gives a residual control input that corrects the model-based controller.

## Repository layout

```
├── runs/                    # raw Aqua2 runs, one folder per run
├── data/results_xy.xlsx     # success probabilities: real + synthetic targets
├── make_results_xy.py       # runs/  ->  data/results_xy.xlsx
├── Trainning.ipynb          # trains the network and plots the results
├── nn_model_synthetic_trajectories.keras   # trained network used in the paper
└── requirements.txt
```

## Raw runs (`runs/`)

Each folder is one run. Its name encodes the target and the start time:

| Folder | Target $`(x_d, y_d, \theta_d)`$ |
|---|---|
| `test_60_2026_06_16_06_55` | (6, 0, 0) |
| `test_3n3pi4_...` | (3, −3, +π/4) |
| `test_93npi4_...` | (9, 3, −π/4) |

The pattern is `<x><n if y<0><|y|><pi4 | npi4 | nothing for 0>`. Targets cover $`x_d \in \{3,6,9\}`$ m, $`y_d \in \{-3,0,3\}`$ m and $`\theta_d \in \{-\pi/4, 0, \pi/4\}`$.

Each folder contains `.npy` arrays sampled at 10 Hz. The ones used for scoring are:

| File | Content |
|---|---|
| `timestamps.npy` | time [s] |
| `traj_x.npy`, `traj_y.npy` | position [m] |
| `head_meas.npy` | heading from the IMU [rad] |

The other files hold extra measurements and controller commands: `vx_meas`, `vy_meas`, `ax_meas`, `ay_meas`, `ang_vel`, `u_command`, `r_command` and `control`.

## Success criterion

A run is successful if, at some time within 60 s, both of these hold at the same moment:

```math
\sqrt{(x-x_d)^2+(y-y_d)^2}\le 0.3\ \text{m}, \qquad |\mathrm{wrap}(\theta-\theta_d)|\le 20^\circ .
```

## Synthetic data

To enlarge the small field dataset, `make_results_xy.py` creates targets at 25%, 50% and 75% of the way between neighbouring real targets along $`x`$ or $`y`$. For each new target, it blends matched pairs of real runs from the two neighbours and scores the result with the same criterion. This gives 26 real and 102 synthetic targets.

```bash
python3 make_results_xy.py runs/ data/results_xy.xlsx
```

## Training

`Trainning.ipynb` loads `data/results_xy.xlsx` and applies Laplace smoothing, $`p = (k+1)/(n+2)`$. It then trains a 3-input network (2 × 32 `tanh` units, sigmoid output) on the synthetic targets and tests it on the real ones. Finally, it loads the trained model and plots the predicted success probability at $`\theta_d = -\pi/4`$.

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@article{faros_tanner_aqua2,
  title  = {Neural-based feedback control on an Autonomous Surface Vehicle},
  author = {Faros, Ioannis and Tanner, Herbert G.},
  journal = {Frontiers},
  note   = {Under review}
}
```

Funded in part by the USGS NGWOS R&D program, award #G23AC00149-00. Contact: ifaros@udel.edu
