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
        "n_repetitions": 5,                 # Default: 5
        "UNUSED_CPU_CORES": 2,              # Number of CPU cores to leave unused when using multiprocessing, for appropriate other applications' performance. Cap degree of parallelism: <#total CPU cores>-UNUSED_CPU_CORES Default: 2.
        # Plotting parameters
        "benchmark_curve": 1,               # Default: 1, choose one: 1 or 2 for the benchmark CSV (BaselineDataCartPole_run1.csv or BaselineDataCartPole_run2.csv).
        "benchmark_name": "Baseline",
        "plot_smoothing_window": [1, 101, 201, 251, 351],  # Use multiple values to plot multiple curves. Default: [1, 51, 101, 201, 251, 301]. Set to [1] to skip smoothing.
        "curve_confidence_interval": 0.6,   # Curve shading confidence interval. Default: 0.95. Set to 0 to skip CI shading.
        "curve_shaded_area_opacity": 0.06,  # Opacity of the shaded area for confidence intervals. Default: 0.05 (5% opacity).
        "show_curve_plots": True,                # Show learning curve plot window at the end.
        "animation_plot": True,            # Show CartPole animation at the end.
        "use_existing_disk_data": True,     # Whether to use existing data (.xlsx files) from disk if exists.
        "use_existing_disk_trained_networks": True,
        # Environment
        "max_train_episode_length": 1000, #500        # Episode truncation step. Default: 500.
        "base_seed": 42,                    # Base seed for CartPole environment and agent initialization. Each repetition will use a different seed derived from this base seed (e.g., base_seed + repetition_index).
        # Agent
        "n_timesteps": 200000,              # Total number of training timesteps. Default: 1000000.
        "max_eval_episode_length": 20000, #500        # Episode truncation step. Default: 500.
        "eval_interval": 250,
        "eval_with_env_episode_trials": True, # Default: True. Set to False to use the fast proxy from training (last_episode_return) for evaluation instead of running separate greedy environment episode trials via agent.evaluate(). Note: setting to False will speed up training and plotting, but will not provide true evaluation curves. Setting to True will provide true evaluation curves but will significantly increase training time due to the need to run separate evaluation episodes at each eval_interval.
        "n_eval_episodes": 5,
    }
    ################[ End Global Parameters ]################


    ###############[ Training hyperparameters ]##############
    # ------------- Algorithm Type: REINFORCE hyperparameters (optimal) ----
    include_REINFORCE_in_training = False
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
    include_A2C_in_training = True
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

    # ------------- Algorithm Type: PPO hyperparameters (optimal) ----
    # Proximal Policy Optimisation (PPO-clipped) - Schulman et al., 2017
    include_PPO_in_training = True
    ppo_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [0.0003],           # actor learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],  # actor NN architectures to sweep
        "critic_lr": [0.001],           # critic learning rate(s) to sweep
        "critic_hidden_nn": [[64, 64]], # critic NN architectures to sweep
        "gae_lambda": [0.95],           # GAE lambda (Schulman et al., 2015)
        "clip_eps": [0.2],              # PPO clipping epsilon
        "n_epochs": [10],               # # of optimisation epochs per rollout
        "rollout_steps": [2048],        # # of env steps per rollout (PPO buffer size)
        "mini_batch_size": [64],        # mini-batch size for PPO updates
        "entropy_coef": [0.0],          # entropy bonus weight
        "value_coef": [0.5],            # value loss weight
        "max_grad_norm": [0.5],         # global gradient-norm clip
        "legend_parameters": {          # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "critic_lr": [r"Critic $\beta$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
            "critic_hidden_nn": [r"Critic NN: ", True],
            "gae_lambda": [r"$\lambda_{GAE}$: ", True],
            "clip_eps": [r"$\epsilon_{clip}$: ", True],
            "n_epochs": [r"Epochs: ", False],
            "rollout_steps": [r"Rollout: ", False],
            "mini_batch_size": [r"MB: ", False],
            "entropy_coef": [r"Ent: ", False],
            "value_coef": [r"VCoef: ", False],
            "max_grad_norm": [r"GradClip: ", False],
        },
    }
    # ------------- End PPO hyperparameters -----------

    # ------------- Algorithm Type: SAC hyperparameters (optimal) ----
    # Soft Actor-Critic (discrete) - Haarnoja et al., 2018/2019 + Christodoulou, 2019
    include_SAC_in_training = True
    sac_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [0.0003],           # actor learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],  # actor NN architectures to sweep
        "critic_lr": [0.0003],          # critic learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]], # critic (twin Q) NN architectures
        "alpha_lr": [0.0003],           # entropy-temperature learning rate
        "tau": [0.005],                 # soft target update rate
        "target_entropy_ratio": [0.98], # target entropy = ratio * log(n_actions)
        "replay_buffer_size": [100000], # replay buffer capacity
        "batch_size": [64],             # SGD batch size from the replay buffer
        "warmup_steps": [1000],         # random-action steps before SAC kicks in
        "updates_per_step": [1],        # gradient updates per env step
        "auto_tune_alpha": [True],      # Default: True (Haarnoja et al., 2019). Set to [False] to use fixed alpha (Haarnoja et al., 2018).
        "alpha_init": [1.0],            # Initial / fixed entropy temperature alpha. Used as the start point when auto_tune_alpha=True, or as the fixed value when auto_tune_alpha=False.
        "legend_parameters": {          # [plot label, show flag]
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "critic_lr": [r"Critic $\beta$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
            "critic_hidden_nn": [r"Critic NN: ", True],
            "alpha_lr": [r"$\alpha_{lr}$: ", False],
            "tau": [r"$\tau$: ", True],
            "target_entropy_ratio": [r"Tgt H ratio: ", False],
            "replay_buffer_size": [r"Buff: ", False],
            "batch_size": [r"Batch: ", False],
            "warmup_steps": [r"Warm: ", False],
            "updates_per_step": [r"UPS: ", False],
            "auto_tune_alpha": [r"AutoTune $\alpha$: ", True],
            "alpha_init": [r"$\alpha_0$: ", False],
        },
    }
    # ------------- End SAC hyperparameters -----------

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
    if include_PPO_in_training:
        experiments.append("PPO")
    if include_SAC_in_training:
        experiments.append("SAC")
    if include_DQN_in_training:
        experiments.append("DQN")

    run_selected_experiments(
        experiments,
        global_config=global_config,
        reinforce_config=reinforce_config,
        ac_config=ac_config,
        a2c_config=a2c_config,
        dqn_config=dqn_config,
        ppo_config=ppo_config,
        sac_config=sac_config,
    )


if __name__ == '__main__':
    experiment()
