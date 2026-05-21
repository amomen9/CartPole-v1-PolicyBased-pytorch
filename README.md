# CartPole-v1 PPO - Implementation and Comparative Study

Repository for the Reinforcement Learning course assignment 4 (Leiden University) - **Proximal Policy Optimisation on CartPole-v1**.

## Overview

This project implements PPO and compares it with multiple other reinforcement learning agents which were implemented before for the [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment:

- **REINFORCE**
- **Advantage Actor-Critic (A2C)**
- **DQN**

The experiment pipeline is driven by `Experiment.py`, which defines the global run settings and algorithm-specific hyperparameters. By default, the script runs **PPO together with DQN and A2C** for the final comparison; the other algorithms can be enabled or disabled by toggling the corresponding flags in the `included_algorithms` dictionary. Dedicated `Experiment_*.py` scripts are provided so that each ablation / sweep can be reproduced with a single command.

## Setup

### Prerequisites

- Python 3
- `pip`
- A working C/C++ build toolchain may be useful for some Python packages, depending on the platform

### Installation

Install the required python packages using the command below:

```bash
pip install -r requirements.txt
```
or
```bash
python -m pip install -r requirements.txt
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

Every result in the report can be reproduced with a **single command** by running `Experiment.py`. Different sub-tasks (the final comparison and each PPO hyperparameter sweep) are selected via the `included_algorithms` dictionary and the lists inside `PPO_config` (and the other algorithm configs) at the top of `Experiment.py`.

### Run the Final Comparison (PPO vs. A2C vs. DQN)

```bash
python Experiment.py
```

This is the main entrypoint and reproduces the final learning-curve comparison between PPO (with its optimal hyperparameters), A2C and DQN. Algorithms can be turned on/off by editing the `included_algorithms` dictionary inside `Experiment.py`.

Alternatively, the algorithms to include can be selected directly from the command line with one flag per algorithm (`--DQN`, `--REINFORCE`, `--AC`, `--A2C`, `--PPO`). When any algorithm flag is provided, it overrides the `included_algorithms` dictionary inside `Experiment.py`. For example, to run all four algorithms together:

```bash
python Experiment.py --A2C --DQN --AC --PPO
```

### Run the PPO Ablation / Hyperparameter Sweeps

Every PPO hyperparameter sweep used in the report is reproduced from `Experiment.py` by extending the corresponding list inside `PPO_config` (for example, set `"gamma": [0.9, 0.99, 1.0]` to sweep three discount factors, or `"clip_eps": [0.1, 0.2, 0.3]` to sweep the clip range), then running the same command:

```bash
python Experiment.py
```

The same applies to actor / critic learning rates, network architectures, `gae_lambda`, `n_epochs`, and `rollout_steps` - each is a list inside `PPO_config` and any combination of values will be swept in a single run.

### Test the Environment

The project is centered on the CartPole-v1 environment. A quick smoke test is to run the main experiment script and confirm the training pipeline starts correctly with the Gymnasium environment and dependencies installed.

### Visualize a Trained Policy

Set the plotting flags in `Experiment.py` to `True`:

- `show_curve_plots = True`
- `animation_plot = True`

This enables learning-curve visualization and optional CartPole animation output.

### Reuse Existing Results

`Experiment.py` exposes two flags in `global_config` that control caching:

- `use_existing_disk_data` - reuse `.xlsx` runs from `data sheets/` if present
- `use_existing_disk_networks_checkpoints` - reuse saved networks from `Checkpoints/`

Setting both to `True` lets a sweep finish in seconds when the data already exists on disk; setting them to `False` forces a fresh training run.

### Customize Experiments

Adjust the configuration dictionaries inside `Experiment.py`:

- `global_config`
- `PPO_config`
- `DQN_config`
- `A2C_config`
- `REINFORCE_config`
- `AC_config`

This is the intended entry point for changing architectures, learning rates, clip ranges, rollout lengths, or benchmark settings.

## Environment

- **Task:** CartPole-v1
- **Framework:** Gymnasium
- **Primary libraries:** NumPy, SciPy, Matplotlib, Pandas, OpenPyXL, tqdm, PyTorch
- **Output style:** plots, spreadsheets, and optional animations
- **Default benchmark source:** `benchmark_curve = 1` with benchmark name `Baseline`

The project is configured to reuse existing disk data when available, which keeps repeated experiment runs fast and reproducible.

## Project Structure

- `Experiment.py` - main experiment entrypoint and hyperparameter configuration (final PPO vs. A2C vs. DQN comparison and all PPO sweeps)
- `PPO.py` - PPO-clipped (with GAE) implementation
- `Library.py` - central library that wires experiments to algorithms, plotting, and Excel logging
- `Helper.py` - shared helper functions
- `Agent.py` - agent logic shared across algorithms
- `Environment.py` - environment wrapper / interaction logic
- `Checkpointing.py` - utilities for saving/loading trained networks
- `REINFORCE.py`, `AC.py`, `A2C.py` - reused policy-gradient / actor-critic implementations from the previous assignment
- `assignment2_repo/` - previous assignment code reused for DQN
- `assignment3_repo/` - previous assignment code reused for REINFORCE / AC / A2C baselines
- `Baseline data/` - benchmark CSV files used as the reference learning curve
- `Checkpoints/` - saved trained networks reused when `use_existing_disk_networks_checkpoints = True`
- `data sheets/` - `.xlsx` run logs reused when `use_existing_disk_data = True`
- `plots/` - generated learning-curve figures
- `README.md` - this file
- `requirements.txt` - Python dependencies

## Architecture & Implementation Details

### Neural Network Modules

The PPO actor (categorical policy) and critic (state-value) networks are compact multilayer perceptrons tuned for CartPole-v1. The baseline implementations reused for comparison use the same MLP topology pattern:

- **PPO:** actor network `[128, 128]`, critic network `[512, 512]`
- **A2C:** actor network `[64, 64]`, critic network `[128, 128]`
- **REINFORCE:** actor network `[128, 128]`
- **DQN:** Q-network `[128, 128]`

### Training Loop

PPO collects fixed-length rollouts (`rollout_steps` env transitions), computes GAE advantages, and then performs `n_epochs` full-batch gradient updates of the clipped surrogate objective per rollout. The shared experiment pipeline coordinates repeated runs, logging, smoothing, and benchmarking. The training loop is parameterised through the dictionaries in `Experiment.py`, so experiments can be reproduced or swept with minimal code changes.

### Generalized Advantage Estimation (GAE)

PPO uses GAE (Schulman et al., 2015) as the advantage estimator:

- `gae_lambda = 0.96` (bias-variance trade-off)
- Setting `gae_lambda = 1.0` falls back to plain Monte-Carlo advantages

### PPO-clipped Surrogate Objective

The clipped surrogate objective constrains the policy ratio between the new and old policies:

- `clip_eps = 0.1` (clipping range for the probability ratio)
- `n_epochs = 30` (gradient passes over each rollout)
- `rollout_steps = 512` (env steps per rollout / PPO buffer size)

### Target Network

Target network updates are used where applicable for the baselines:

- **A2C:** `TN_step = 10`
- **DQN:** `TN_active = True`, `TN_step = 100`

PPO does not use a target network; policy stability is enforced through the clipped surrogate objective.

### Experience Replay

Experience replay is enabled for DQN only:

- `ER_active = True`
- `ER_replay_buffer_size = 80000`
- `ER_batch_size = 64`
- `ER_min_replay_size = 2000`
- `ER_sample_train_frequency = 1`
- `ER_replay_ratio = 1.0`

PPO is on-policy, so its rollout buffer is fully refreshed between updates.

### Exploration Strategies

PPO explores implicitly through the stochasticity of its categorical policy. DQN uses epsilon-greedy exploration by default:

- `exploration_method = "egreedy"`
- `epsilons = [0.05]`
- `epsilon_start = 0.05`
- `epsilon_end = 0.05`
- `epsilon_decay = 1`
- `epsilon_decay_interval = 0`

A softmax exploration sweep is also defined for DQN:

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

| Parameter                              |                    Value |
| -------------------------------------- | -----------------------: |
| `n_repetitions`                      |                    `5` |
| `plot_smoothing_window`              | `[101, 201, 251, 351]` |
| `curve_confidence_interval`          |                  `0.6` |
| `curve_shaded_area_opacity`          |                 `0.06` |
| `use_existing_disk_data`             |                `False` |
| `use_existing_disk_networks_checkpoints` |                 `True` |
| `show_curve_plots`                   |                 `True` |
| `animation_plot`                     |                `False` |
| `n_timesteps`                        |              `1000000` |
| `eval_interval`                      |                  `250` |
| `max_train_episode_length`           |                  `500` |
| `max_eval_episode_length`            |                  `500` |
| `n_eval_episodes`                    |                    `5` |
| `base_seed`                          |                   `42` |

### PPO (Optimal)

| Parameter            |            Value |
| -------------------- | ---------------: |
| `gamma`            |       `[0.99]` |
| `actor_lr`         |       `[3e-4]` |
| `actor_hidden_nn`  | `[[128, 128]]` |
| `critic_lr`        |       `[0.01]` |
| `critic_hidden_nn` | `[[512, 512]]` |
| `gae_lambda`       |       `[0.96]` |
| `clip_eps`         |        `[0.1]` |
| `n_epochs`         |         `[30]` |
| `rollout_steps`    |        `[512]` |

### REINFORCE

| Parameter           |            Value |
| ------------------- | ---------------: |
| `gamma`           |       `[0.99]` |
| `actor_lr`        |       `[1e-3]` |
| `actor_hidden_nn` | `[[128, 128]]` |

### Advantage Actor-Critic (A2C)

| Parameter            |            Value |
| -------------------- | ---------------: |
| `gamma`            |       `[0.99]` |
| `actor_lr`         |       `[1e-4]` |
| `actor_hidden_nn`  |     `[64, 64]` |
| `critic_lr`        |       `[0.01]` |
| `critic_hidden_nn` | `[[128, 128]]` |
| `TN_step`          |         `[10]` |

### DQN

| Parameter                     |               Value |
| ----------------------------- | ------------------: |
| `gamma`                     |            `0.99` |
| `learning_rate`             |         `[0.001]` |
| `nn_hidden_layer_widths`    |    `[[128, 128]]` |
| `exploration_method`        |       `"egreedy"` |
| `epsilons`                  |          `[0.05]` |
| `epsilon_start`             |            `0.05` |
| `epsilon_end`               |            `0.05` |
| `epsilon_decay`             |               `1` |
| `epsilon_decay_interval`    |               `0` |
| `softmax_temps`             | `[1.0, 0.5, 0.1]` |
| `TN_active`                 |          `[True]` |
| `TN_step`                   |           `[100]` |
| `ER_active`                 |          `[True]` |
| `ER_replay_buffer_size`     |           `80000` |
| `ER_batch_size`             |            `[64]` |
| `ER_min_replay_size`        |            `2000` |
| `ER_sample_train_frequency` |             `[1]` |
| `ER_replay_ratio`           |             `1.0` |

## Ablation Study Results

The repository is set up to generate ablation results automatically from the configuration in `Experiment.py`. Every PPO sub-task in the report (gamma, actor LR, critic LR, actor NN, critic NN, clip epsilon, optimisation epochs) is reproduced by editing the corresponding list inside `PPO_config` and rerunning `python Experiment.py` as described in the [Usage](#usage) section.

- Learning curves are produced from repeated runs.
- Existing `.xlsx` data can be reused from disk when `use_existing_disk_data = True`.
- Plots and optional animations are generated through the shared experiment pipeline.

If you rerun the experiments with different hyperparameters, the resulting curves and spreadsheet logs can be compared directly against the benchmark configuration.

## Key Design Decisions

- **Single configuration entrypoint:** all important run settings live in `Experiment.py`, so every ablation / sweep is reproduced with a single command
- **Algorithm separation:** each method has its own module for clarity and maintainability
- **Shared experiment tooling:** common logging, plotting, and Excel handling are centralised in `Library.py`
- **Legacy baseline reuse:** the DQN implementation from assignment 2 and the REINFORCE / AC / A2C implementations from assignment 3 are preserved in `assignment2_repo/` and `assignment3_repo/`
- **Disk caching:** `use_existing_disk_data` and `use_existing_disk_networks_checkpoints` let sweeps be re-plotted without retraining
- **Reproducibility:** repeated runs use a fixed base seed and a consistent experiment structure

## License

No explicit license is provided in the repository. Treat the code as course assignment material unless a license is added later.
