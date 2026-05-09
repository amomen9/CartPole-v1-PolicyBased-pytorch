#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from functions import run_selected_experiments


# ── Main experiment ───────────────────────────────────────────────────────────

def experiment():

    #################[ Global Parameters ]################
    global_config = {
        "benchmark_curve": 1,               # Default: 1, choose one: 1 or 2 for the benchmark CSV (BaselineDataCartPole_run1.csv or BaselineDataCartPole_run2.csv).
        "benchmark_name": "Baseline",
        "n_repetitions": 5,                 # Default: 5
        # Plotting parameters
        "plot_smoothing_window": [1, 101, 201, 251, 351],  # Use multiple values to plot multiple curves. Default: [1, 51, 101, 201, 251, 301]. Set to [1] to skip smoothing.
        "curve_confidence_interval": 0.6,   # Curve shading confidence interval. Default: 0.95. Set to 0 to skip CI shading.
        "curve_shaded_area_opacity": 0.05,  # Opacity of the shaded area for confidence intervals. Default: 0.05 (5% opacity).
        "use_existing_disk_data": False,     # Whether to use existing data (.xlsx files) from disk if exists.
        "curve_plot": False,                # Show learning curve plot window at the end.
        "animation_plot": False,            # Show CartPole animation at the end.
        # Environment
        "n_timesteps": 100000,              # Total number of training timesteps. Default: 1000000.
        "eval_interval": 250,
        "max_train_episode_length": 500, #500        # Episode truncation step. Default: 500.
        "max_eval_episode_length": 500, #500        # Episode truncation step. Default: 500.
        # Model evaluation option:
        # False (default): evaluation uses the fast proxy from training (last_episode_return).
        # True: evaluation uses greedy environment episode trials via agent.evaluate().
        "eval_with_env_episode_trials": False,
        "n_eval_episodes": 5,
        "base_seed": 42,                    # Base seed for CartPole environment and agent initialization. Each repetition will use a different seed derived from this base seed (e.g., base_seed + repetition_index).
    }
    ################[ End Global Parameters ]################


    ###############[ Training hyperparameters ]##############
    # ------------- Algorithm Type: REINFORCE hyperparameters (optimal) ----
    include_REINFORCE_in_training = True
    reinforce_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [0.001],            # actor learning rate(s) to sweep
        "actor_hidden_nn": [[32,32]],   # list of NN architectures to sweep
        "legend_parameters": {          # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha:$ ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
        },
    }
    # ------------- End REINFORCE hyperparameters -----------

    # ------------- Algorithm Type: AC hyperparameters (optimal) ----
    include_AC_in_training = False
    ac_config = {
        "gamma": [0.99],
        "actor_lr": [0.001],                   # 2D actor learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],         # 2D list of NN architectures to sweep
        "critic_lr": [0.001],                  # critic learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]],      # critic NN architectures to sweep
        "TN_step": [10],                 # list of n-step returns to sweep (Target Network). Default: [10]. Set to [1] to skip n-step return trials.    
        "legend_parameters": {                 # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
            "critic_lr": [r"Critic $\beta$: ", True],
            "critic_hidden_nn": [r"Critic NN: ", True],
            "TN_step": [r"TN Step: ", False],
        },
    }
    # ------------- End AC hyperparameters -----------

    # ------------- Algorithm Type: A2C hyperparameters (optimal) ----
    include_A2C_in_training = False
    a2c_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [0.0001],          # policy learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],        # list of NN architectures to sweep for policy network
        "critic_lr": [0.01],        # value function learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]],  # list of NN architectures to sweep for value function network
        "TN_step": [10],                 # list of n-step returns to sweep (Target Network). Default: [10]. Set to [1] to skip n-step return trials.    
        "legend_parameters": {          # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "critic_lr": [r"Critic $\beta$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
            "critic_hidden_nn": [r"Critic NN: ", True],
            "TN_step": [r"TN Step: ", False],
        },
    }
    # ------------- End A2C hyperparameters -----------

    # Using DQN implementation from the previous assignment (existing in the assignment2_repo directory)
    # ------------- Algorithm: DQN hyperparameters (optimal) ----
    include_DQN_in_training = False
    dqn_config = {
        "gamma": 0.99,
        "learning_rate": [0.001],
        "nn_hidden_layer_widths": [[128, 128]],
        "exploration_method": "egreedy",
        "epsilons": [0.05],
        "epsilon_start": 0.05,
        "epsilon_end": 0.05,
        "epsilon_decay": 1,             #0.9995,
        "epsilon_decay_interval": 0,    #0,    # Default: 0, Optimal: 1 - Set to 0 to skip epsilon decay trials.
        "softmax_temps": [1.0, 0.5, 0.1],
        "TN_active": [True],
        "TN_step": [100],
        "ER_active": [True],
        "ER_replay_buffer_size": 80000,
        "ER_batch_size": [64],
        "ER_min_replay_size": 2000,
        "ER_sample_train_frequency": [1],
        "ER_replay_ratio": 1.0,
        "legend_parameters": {              # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "learning_rate": [r"$\alpha$: ", True],
            "nn_hidden_layer_widths": [r"NN Widths: ", True],
            "exploration_method": [r"Exp Method: ", False],
            "epsilons": [r"$\epsilon$: ", True],
            "epsilon_start": [r"$\epsilon$ Start: ", False],
            "epsilon_end": [r"$\epsilon$ End: ", False],
            "epsilon_decay": [r"$\epsilon$ decay: ", False],
            "epsilon_decay_interval": [r"$\epsilon$ decay int: ", False],
            "softmax_temps": [r"softmax $\tau$: ", False],
            "TN_active": [r"TN: ", True],
            "TN_step": [r"TN Step: ", False],
            "ER_active": [r"ER: ", True],
            "ER_replay_buffer_size": [r"Buff Size: ", False],
            "ER_batch_size": [r"Batch Size: ", False],
            "ER_min_replay_size": [r"Min Rep: ", False],
            "ER_sample_train_frequency": [r"Train Freq: ", False],
            "ER_replay_ratio": [r"Rep Ratio: ", False],
        },
    }
    # ------------- End DQN hyperparameters -----------
    ##########################################################

    experiments = []
    if include_REINFORCE_in_training:
        experiments.append("REINFORCE")
    if include_AC_in_training:
        experiments.append("AC")
    if include_A2C_in_training:
        experiments.append("A2C")
    if include_DQN_in_training:
        experiments.append("DQN")

    run_selected_experiments(
        experiments,
        global_config=global_config,
        reinforce_config=reinforce_config,
        ac_config=ac_config,
        a2c_config=a2c_config,
        dqn_config=dqn_config,
    )


if __name__ == '__main__':
    experiment()
