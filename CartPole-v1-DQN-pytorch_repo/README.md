# CartPole-v1 Deep Q-Network (DQN) — Ablation Study

Repository for the Reinforcement Learning course assignment (Leiden University) — **Deep Q-Learning on CartPole-v1**.

## Overview

This project implements a **Deep Q-Network (DQN)** agent using PyTorch to solve the [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. The core focus is an **ablation study** that systematically evaluates the contribution of two key DQN components:

| Configuration       | Target Network (TN) | Experience Replay (ER) |
| ------------------- | :-----------------: | :--------------------: |
| **Naive DQN** |         ✗         |           ✗           |
| **TN only**   |         ✓         |           ✗           |
| **ER only**   |         ✗         |           ✓           |
| **TN + ER**   |         ✓         |           ✓           |

The agent is trained across multiple repetitions (seeds) with configurable hyperparameters, and results are compared against a pre-recorded baseline benchmark. Learning curves are plotted with confidence intervals and optional Savitzky–Golay smoothing at multiple window sizes.

## Environment

**CartPole-v1** is a classic control task where a pole is attached to a cart moving along a frictionless track. The agent must keep the pole balanced by applying left/right forces.

- **State space**: 4 continuous values — cart position, cart velocity, pole angle, pole angular velocity
- **Action space**: 2 discrete actions — push left (0), push right (1)
- **Reward**: +1 for every timestep the pole remains upright
- **Optimal return**: 500 (episode truncation limit)
- **Termination**: pole angle > ±12° or cart position > ±2.4

## Project Structure

```
CartPole-v1/
├── mylibrary.py              # Core DQN implementation (PolicyNetwork, training loop, replay buffer)
├── dqn_agent.py              # Trial runner — wraps a single DQN training run
├── Experiment.py             # Main experiment orchestrator — ablation study with parallelization
├── Environment.py            # CartPole-v1 wrapper, env factory, animation utilities
├── Helper.py                 # Plotting (LearningCurvePlot), smoothing, ε-greedy, softmax, annealing
├── requirements.txt          # Python dependencies
├── BaselineDataCartPole_run1.csv   # Benchmark baseline curve (run 1)
├── BaselineDataCartPole_run2.csv   # Benchmark baseline curve (run 2)
├── LICENSE                   # MIT License
│
├── plots/                    # Generated learning curve plots (current experiment)
├── data sheets/              # Saved experiment results in Excel format (.xlsx)
├── ablation study graphs/    # Organized ablation study results
│   ├── All/                  #   Full ablation (all configurations combined, 1M steps)
│   ├── TN/                   #   Target Network only ablation
│   ├── ER/                   #   Experience Replay only ablation
│   ├── Multiple TN/          #   Multiple TN step interval comparisons
│   └── Mutiple ER/           #   Multiple ER hyperparameter comparisons
│       └── Sampling Frequency/  #   ER sampling frequency ablation
└── results history/          # Historical experiment plots across different configurations
    └── Plots/
```

## Architecture & Implementation Details

### Neural Network (`mylibrary.py` → `PolicyNetwork`)

The Q-network is a fully-connected feedforward neural network built with `nn.Sequential`:

- **Input layer**: 4 neurons (state dimension)
- **Hidden layers**: configurable widths (default: **128 × 128**), each followed by **ReLU** activation
- **Output layer**: 2 neurons (Q-values for each action)
- **Optimizer**: Adam

### DQN Training Loop (`mylibrary.py` → `train_dqn`)

The training loop runs for a configurable number of environment steps in a single stream using Gymnasium's `Autoreset` wrapper:

1. Select action via the chosen exploration strategy
2. Execute action, observe reward and next state
3. **Without ER**: perform an immediate single-transition Q-learning update (`dqn_update`)
4. **With ER**: store transition in replay buffer; when buffer has enough samples, perform a batch update (`dqn_batch_update`) at the configured sampling frequency
5. Periodically record episode returns at the evaluation interval
6. Decay epsilon (if applicable) at the configured decay interval

### Target Network

When active, a separate **target network** (deep copy of the online network) provides bootstrap Q-value targets. It is synchronized with the online network every `TN_step` environment steps (default: 100). This stabilizes training by reducing the correlation between the target and the current Q-estimates.

### Experience Replay (`mylibrary.py` → `ReplayBuffer`)

A circular buffer stores `(state, action, reward, next_state, terminated)` transitions. Key parameters:

| Parameter                     | Default | Optimal | Description                                     |
| ----------------------------- | ------- | ------- | ----------------------------------------------- |
| `er_replay_buffer_size`     | 10,000  | 80,000  | Maximum buffer capacity                         |
| `er_batch_size`             | 64      | 64      | Mini-batch size for updates                     |
| `er_min_replay_size`        | 100     | 2,000   | Minimum transitions before training begins      |
| `er_sample_train_frequency` | 1       | 1       | Train every N env steps                         |
| `er_replay_ratio`           | 1.0     | 1.0     | Number of gradient updates per training trigger |

### Exploration Strategies

Two exploration strategies are supported:

1. **ε-greedy** (default): with exponential decay from `epsilon_start` (1.0) → `epsilon_end` (0.02) using `epsilon_decay` (0.9995)
2. **Softmax (Boltzmann)**: action probabilities proportional to $e^{Q(s,a)/\tau}$ with configurable temperature $\tau$

### Experiment Orchestrator (`Experiment.py`)

The `experiment()` function:

- Constructs a Cartesian product of all hyperparameter combinations (TN on/off, ER on/off, batch sizes, architectures, exploration parameters, learning rates, etc.)
- Runs each combination for `n_repetitions` seeds (default: 5) with parallel execution via `ProcessPoolExecutor`
- Supports **caching**: saves results to Excel (`.xlsx`) in `data sheets/` and reloads them on subsequent runs to avoid re-running experiments
- Plots learning curves with multiple smoothing windows (1, 51, 101, 201, 251, 301) alongside the benchmark baseline and the CartPole optimum line (500)
- Adds **confidence interval shading** around the mean curves

## Hyperparameters (Optimal Configuration)

| Hyperparameter          | Value       |
| ----------------------- | ----------- |
| Discount factor (γ)    | 0.99        |
| Learning rate           | 0.001       |
| Network architecture    | [128, 128]  |
| Max episode length      | 500         |
| Evaluation interval     | 250 steps   |
| Training timesteps      | 220,000     |
| Repetitions (seeds)     | 5           |
| Base seed               | 42          |
| ε-greedy ε (constant) | 0.05        |
| ε-decay: start → end  | 1.0 → 0.02 |
| ε-decay rate           | 0.9995      |
| TN update interval      | 100         |
| ER buffer size          | 80,000      |
| ER batch size           | 64          |
| ER min replay size      | 2,000       |
| ER sample frequency     | 1           |

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Dependencies

| Package                    | Minimum Version |
| -------------------------- | --------------- |
| numpy                      | ≥ 1.26         |
| scipy                      | ≥ 1.11         |
| matplotlib                 | ≥ 3.8          |
| pandas                     | ≥ 2.2          |
| openpyxl                   | ≥ 3.1          |
| tqdm                       | ≥ 4.66         |
| gymnasium[classic-control] | ≥ 1.0          |
| torch                      | ≥ 2.2          |
| pillow                     | ≥ 10.0         |

## Usage

### Run the Full Experiment

```bash
python Experiment.py
```

This executes the ablation study with the parameters defined at the top of `Experiment.py`. Results are saved to `data sheets/` as Excel files and plots are saved to `plots/`. On subsequent runs, if `use_existing_disk_data = True` (default), cached results are loaded from disk instead of re-running.

### Test the Environment

```bash
python Environment.py
```

Runs a quick random-action test (100 steps) on CartPole-v1 and writes log output to `output_Environment.py.log`.

### Visualize a Trained Policy

```python
from Environment import show_one_episode
from dqn_agent import neural_net_policy, run_dqn_trial

# Train an agent
model, returns, timesteps = run_dqn_trial(
    n_env_steps=100000,
    max_episode_length=500,
    learning_rate=0.001,
    nn_hidden_layer_widths=[128, 128],
    discount_factor=0.99,
    target_network_step=100,
    n_returns_interval=250,
    exploration_method="epsilon_greedy",
    epsilon_start=1.0,
    epsilon_end=0.02,
    epsilon_decay=0.9995,
    er_active=True,
    er_replay_buffer_size=80000,
    er_batch_size=64,
    er_min_replay_size=2000,
    seed=42,
)

# Animate the trained agent
anim = show_one_episode(lambda obs: neural_net_policy(obs, model))
```

### Customize Experiments

Edit the parameters at the top of `Experiment.py` to:

- Toggle Target Network and Experience Replay (`TN_active`, `ER_active`)
- Compare multiple ER batch sizes, sampling frequencies, or TN update intervals
- Switch between ε-greedy and softmax exploration
- Configure PyTorch compilation/runtime (`compilation_config`) including `torch.compile`, backend fallback, TorchScript mode, and matmul precision
- Adjust smoothing windows and confidence interval shading
- Control parallelization (auto-detected based on CPU count)

### PyTorch Compilation Settings

The experiment now includes a `compilation_config` dictionary in `Experiment.py` that is propagated to every parallel trial process:

- `enable_torch_compile`: enables `torch.compile` (PyTorch 2.x). Default is `False` for stable performance across environments.
- `torch_compile_backend`: primary backend (default: `inductor`)
- `torch_compile_fallback_backend`: optional fallback backend if the primary backend fails (default: `None`; set `aot_eager` if desired)
- `torch_compile_mode`: optimization mode for the `inductor` backend
- `torch_compile_prewarm`: runs one-time warm-up forwards before training to shift compile latency out of the step loop
- `reuse_compiled_model_across_trials`: reuses compiled model wrappers across trials with the same architecture/config in a worker process
- `torchscript_train_mode`: optional TorchScript compile path (`none`, `script`, `trace`)
- `matmul_precision`: float32 matmul precision (`highest`, `high`, `medium`)
- `enable_cuda_tf32`: enables TF32 kernels on CUDA devices

On systems where `inductor` cannot compile C++ kernels (for example, missing OpenMP headers on Windows), keeping `enable_torch_compile=False` is usually fastest; you can enable it explicitly after setting up the required toolchain.

Note: with multiprocessing, compilation caches are process-local, so reuse is "once per worker process" rather than literally once for the entire OS project run.

## Ablation Study Results

The `ablation study graphs/` directory contains pre-computed results organized by configuration:

| Directory                          | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| `All/`                           | All four configurations (Naive, TN, ER, TN+ER) compared over 1M steps |
| `TN/`                            | Target Network on vs. off                                             |
| `ER/`                            | Experience Replay on vs. off                                          |
| `Multiple TN/`                   | Various TN update intervals compared                                  |
| `Mutiple ER/`                    | Various ER hyperparameter settings compared                           |
| `Mutiple ER/Sampling Frequency/` | ER sampling frequency ablation                                        |

Each directory contains an Excel workbook (`.xlsx`) with raw data and PNG plots at multiple smoothing window sizes (non-smoothed, w51, w101, w201, w251, w301, etc.).

## Key Design Decisions

- **Autoreset wrapper**: the training loop uses Gymnasium's `Autoreset` wrapper to maintain a continuous step stream without manual episode resets
- **Parallel execution**: experiments are parallelized across seeds and hyperparameter settings using `concurrent.futures.ProcessPoolExecutor`
- **Deterministic seeding**: each repetition uses `base_seed + rep` for reproducibility via `torch.manual_seed` and `random.seed`
- **Savitzky–Golay smoothing**: learning curves are smoothed with `scipy.signal.savgol_filter` (polynomial order 2) for visualization
- **Confidence intervals**: shaded bands computed using the Student's t-distribution at a configurable significance level (default: 60%)
- **Benchmark comparison**: a pre-recorded baseline curve (CSV) is plotted alongside experimental results for reference

## License

MIT License — see [LICENSE](LICENSE) for details.
