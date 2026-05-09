# CartPole-v1 REINFORCE, Actor-Critic, A2C, and DQN — Comparative Study

Repository for the Reinforcement Learning course assignment (Leiden University) — **policy-gradient and actor-critic methods on CartPole-v1**.

## Overview

This project implements and compares multiple reinforcement learning agents for the [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment:

- **REINFORCE**
- **Actor-Critic (AC)**
- **Advantage Actor-Critic (A2C)**
- **DQN** reuse from the previous assignment

The experiment pipeline is driven by `Experiment.py`, which defines the global run settings and algorithm-specific hyperparameters. By default, the script runs **REINFORCE only**; the other algorithms can be enabled by toggling the corresponding `include_*_in_training` flags.

## Environment

- **Task:** CartPole-v1
- **Framework:** Gymnasium
- **Primary libraries:** NumPy, SciPy, Matplotlib, Pandas, OpenPyXL, tqdm, PyTorch
- **Output style:** plots, spreadsheets, and optional animations
- **Default benchmark source:** `benchmark_curve = 1` with benchmark name `Baseline`

The project is configured to reuse existing disk data when available, which keeps repeated experiment runs fast and reproducible.

## Project Structure

- `Experiment.py` — main experiment entrypoint and hyperparameter configuration
- `functions.py` — core experiment execution utilities
- `facilitation_functions.py` — result extraction, Excel helpers, and job-building utilities
- `Helper.py` — shared helper functions
- `Agent.py` — agent logic shared across algorithms
- `Environment.py` — environment wrapper / interaction logic
- `REINFORCE.py` — REINFORCE implementation
- `AC.py` — actor-critic implementation
- `A2C.py` — advantage actor-critic implementation
- `assignment2_repo/` — previous assignment code reused for DQN
- `Assignment_3.pdf` — assignment statement
- `requirements.txt` — Python dependencies
- `A2C algorithm from book.png`, `AC algorithm from book.png`, `REINFORCE algorithm from book.png` — reference figures
- `test_anim.gif` — example animation artifact

## Architecture & Implementation Details

### Neural Network Modules

The policy and value function approximators are defined per algorithm and use compact multilayer perceptrons tuned for CartPole-v1:

- **REINFORCE:** actor network with hidden layers `[32, 32]`
- **AC:** actor network `[64, 64]`, critic network `[128, 128]`
- **A2C:** actor network `[64, 64]`, critic network `[128, 128]`
- **DQN:** Q-network `[128, 128]`

### Training Loop

Each algorithm module handles its own learning updates, while the shared experiment pipeline coordinates repeated runs, logging, smoothing, and benchmarking. The training loop is parameterized through the dictionaries in `Experiment.py`, so experiments can be reproduced or swept with minimal code changes.

### Target Network

Target network updates are used where applicable:

- **AC:** `TN_step = 10`
- **A2C:** `TN_step = 10`
- **DQN:** `TN_active = True`, `TN_step = 100`

### Experience Replay

Experience replay is enabled for DQN:

- `ER_active = True`
- `ER_replay_buffer_size = 80000`
- `ER_batch_size = 64`
- `ER_min_replay_size = 2000`
- `ER_sample_train_frequency = 1`
- `ER_replay_ratio = 1.0`

### Exploration Strategies

DQN uses epsilon-greedy exploration by default:

- `exploration_method = "egreedy"`
- `epsilons = [0.05]`
- `epsilon_start = 0.05`
- `epsilon_end = 0.05`
- `epsilon_decay = 1`
- `epsilon_decay_interval = 0`

A softmax exploration sweep is also defined:

- `softmax_temps = [1.0, 0.5, 0.1]`

### Experiment Orchestrator

`Experiment.py` is the single source of truth for:

- global run settings
- which algorithms are included in the run
- per-algorithm hyperparameter sweeps
- plotting / smoothing configuration
- benchmark selection
- reproducibility seeds

## Hyperparameters (Optimal Configuration)

### Global Parameters

| Parameter | Value |
|---|---:|
| `benchmark_curve` | `1` |
| `benchmark_name` | `Baseline` |
| `n_repetitions` | `5` |
| `plot_smoothing_window` | `[1, 101, 201, 251, 351]` |
| `curve_confidence_interval` | `0.6` |
| `curve_shaded_area_opacity` | `0.05` |
| `use_existing_disk_data` | `True` |
| `curve_plot` | `False` |
| `animation_plot` | `False` |
| `n_timesteps` | `1000000` |
| `eval_interval` | `250` |
| `max_episode_length` | `500` |
| `base_seed` | `42` |

### REINFORCE

| Parameter | Value |
|---|---:|
| `gamma` | `[0.99]` |
| `actor_lr` | `[0.001, 0.0005, 0.0001]` |
| `actor_hidden_nn` | `[[32, 32]]` |

### Actor-Critic (AC)

| Parameter | Value |
|---|---:|
| `gamma` | `[0.99]` |
| `actor_lr` | `[0.001]` |
| `actor_hidden_nn` | `[[64, 64]]` |
| `critic_lr` | `[0.001]` |
| `critic_hidden_nn` | `[[128, 128]]` |
| `TN_step` | `[10]` |

### Advantage Actor-Critic (A2C)

| Parameter | Value |
|---|---:|
| `gamma` | `[0.99]` |
| `actor_lr` | `[0.0001]` |
| `actor_hidden_nn` | `[[64, 64]]` |
| `critic_lr` | `[0.01]` |
| `critic_hidden_nn` | `[[128, 128]]` |
| `TN_step` | `[10]` |

### DQN

| Parameter | Value |
|---|---:|
| `gamma` | `0.99` |
| `learning_rate` | `[0.001]` |
| `nn_hidden_layer_widths` | `[[128, 128]]` |
| `exploration_method` | `"egreedy"` |
| `epsilons` | `[0.05]` |
| `epsilon_start` | `0.05` |
| `epsilon_end` | `0.05` |
| `epsilon_decay` | `1` |
| `epsilon_decay_interval` | `0` |
| `softmax_temps` | `[1.0, 0.5, 0.1]` |
| `TN_active` | `[True]` |
| `TN_step` | `[100]` |
| `ER_active` | `[True]` |
| `ER_replay_buffer_size` | `80000` |
| `ER_batch_size` | `[64]` |
| `ER_min_replay_size` | `2000` |
| `ER_sample_train_frequency` | `[1]` |
| `ER_replay_ratio` | `1.0` |

## Setup

### Prerequisites

- Python 3
- `pip`
- A working C/C++ build toolchain may be useful for some Python packages, depending on the platform

### Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### Dependencies

The project dependencies are listed in `requirements.txt`:

- `numpy>=1.26`
- `scipy>=1.11`
- `matplotlib>=3.8`
- `pandas>=2.2`
- `openpyxl>=3.1`
- `tqdm>=4.66`
- `gymnasium>=1.0`
- `torch>=2.2`
- `pillow>=10.0`

## Usage

### Run the Full Experiment

Run the experiment entrypoint directly:

```bash
python Experiment.py
```

By default, only `REINFORCE` is enabled in `Experiment.py`. To run other algorithms, set the corresponding flags to `True`:

- `include_AC_in_training`
- `include_A2C_in_training`
- `include_DQN_in_training`

### Test the Environment

The project is centered on the CartPole-v1 environment. A quick smoke test is to run the main experiment script and confirm the training pipeline starts correctly with the Gymnasium environment and dependencies installed.

### Visualize a Trained Policy

Set the plotting flags in `Experiment.py` to `True`:

- `curve_plot = True`
- `animation_plot = True`

This enables learning-curve visualization and optional CartPole animation output.

### Customize Experiments

Adjust the configuration dictionaries in `Experiment.py`:

- `global_config`
- `reinforce_config`
- `ac_config`
- `a2c_config`
- `dqn_config`

This is the intended entry point for changing architectures, learning rates, replay settings, or benchmark settings.

### PyTorch Compilation Settings

No custom compilation settings are required. The project runs in standard eager mode with the installed PyTorch version.

## Ablation Study Results

The repository is set up to generate experiment results automatically from the configuration in `Experiment.py`.

- Learning curves are produced from repeated runs.
- Existing `.xlsx` data can be reused from disk when `use_existing_disk_data = True`.
- Plots and optional animations are generated through the shared experiment pipeline.

If you rerun the experiments with different hyperparameters, the resulting curves and spreadsheet logs can be compared directly against the benchmark configuration.

## Key Design Decisions

- **Single configuration entrypoint:** all important run settings live in `Experiment.py`
- **Algorithm separation:** each method has its own module for clarity and maintainability
- **Shared experiment tooling:** common logging, plotting, and Excel handling are centralized
- **Legacy DQN reuse:** the DQN implementation from assignment 2 is preserved in `assignment2_repo/`
- **Reproducibility:** repeated runs use a fixed base seed and a consistent experiment structure

## License

No explicit license is provided in the repository. Treat the code as course assignment material unless a license is added later.
