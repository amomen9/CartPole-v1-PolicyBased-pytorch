#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np

from Library import run_actor_checkpoint_evaluation_exhaustive


# ── Main experiment ───────────────────────────────────────────────────────────

def experiment():
    
    
    #################[ Global Parameters ]################
    global_config = {
        "UNUSED_CPU_CORES": 4,             # Number of CPU cores to leave unused when using multiprocessing, for appropriate other applications' performance. Cap degree of parallelism: <#total CPU cores>-UNUSED_CPU_CORES Default: 2.
        "plot_smoothing_window": [1,21,51,101],  # Use multiple values to plot multiple curves. Default: [1, 51, 101, 201, 251, 301]. Set to [1] to skip smoothing.
        "show_curve_plots": True,          # Show learning curve plot at the end of or during the training.
        "show_curve_smoothing_windows": [1,21],
        "separate_algorithm_plots": False,  # If True, each algorithm gets its own set of plots (one per smoothing window). Each algo's plots are saved to disk and (if show_curve_plots) shown non-blocking as soon as that algo finishes executing, so faster algos surface their plots first. Default: False (one combined plot per smoothing window).
        "base_seed": 42,                   # Base seed for CartPole environment and agent initialization. Each repetition will use a different seed derived from this base seed (e.g., base_seed + repetition_index).
        "max_eval_episode_length": 20000,    #500        # Episode truncation step. Default: 500.
        "legend_parameters": {              # [plot label, show flag]
            "max_eval_episode_length": [r"EvL: ", False],
        },
    }

    max_eval_episode_length = global_config["max_eval_episode_length"]
    plot_smoothing_window = global_config["plot_smoothing_window"]
    show_curve_smoothing_windows = global_config["show_curve_smoothing_windows"]
    separate_algorithm_plots = global_config["separate_algorithm_plots"]
    show_curve_plots = global_config["show_curve_plots"]

    included_algo_checkpoint_eval = {
        "REINFORCE": {
            "enabled": False,
            "actor_hidden_nn": np.array([128, 128], dtype=np.int32),
        },
        "AC": {
            "enabled": False,
            "actor_hidden_nn": np.array([64, 64], dtype=np.int32),
        },
        "A2C": {
            "enabled": True,
            "actor_hidden_nn": np.array([64, 64], dtype=np.int32),
        },
        "PPO": {
            "enabled": True,
            "actor_hidden_nn": np.array([128, 128], dtype=np.int32),
        },
        "n_episodes": 200,
    }

    run_actor_checkpoint_evaluation_exhaustive(
        included_algo_checkpoint_eval=included_algo_checkpoint_eval,
        max_eval_episode_length=max_eval_episode_length,
        plot_smoothing_window=plot_smoothing_window,
        show_curve_smoothing_windows=show_curve_smoothing_windows,
        separate_algorithm_plots=separate_algorithm_plots,
        show_curve_plots=show_curve_plots,
    )


if __name__ == '__main__':
    experiment()
