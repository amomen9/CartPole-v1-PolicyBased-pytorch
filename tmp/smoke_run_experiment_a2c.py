import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Library import run_selected_experiments  # noqa: E402


if __name__ == "__main__":
    global_config = {
        "benchmark_curve": 1,
        "benchmark_name": "Baseline",
        "n_repetitions": 2,
        "plot_smoothing_window": [1],
        "curve_confidence_interval": 0.0,
        "curve_shaded_area_opacity": 0.05,
        "use_existing_disk_data": False,  # force fresh run to hit the failing path
        "use_existing_network_checkpoints": False,
        "curve_plot": False,
        "animation_plot": False,
        "n_timesteps": 300,
        "eval_interval": 100,
        "max_episode_length": 50,
        "max_train_episode_length": 50,
        "max_eval_episode_length": 50,
        "eval_with_env_episode_trials": False,  # use fast proxy to reduce env rollouts
        "n_eval_episodes": 1,
        "base_seed": 123,
    }

    a2c_config = {
        "gamma": [0.99],
        "actor_lr": [0.001],
        "actor_hidden_nn": [[8]],
        "critic_lr": [0.001],
        "critic_hidden_nn": [[8]],
        "TN_step": [5],
        # legend_parameters omitted intentionally
    }

    run_selected_experiments(
        ["A2C"],
        global_config=global_config,
        a2c_config=a2c_config,
    )
    print("EXPERIMENT_A2C_SMOKE_OK")
