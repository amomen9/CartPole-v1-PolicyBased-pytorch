#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from Library import run_selected_experiments


# ── Main experiment ───────────────────────────────────────────────────────────

def experiment():

    #################[ Global Parameters ]################
    global_config = {
        "n_repetitions": 5,                 # Default: 5
        "UNUSED_CPU_CORES": 4,              # Number of CPU cores to leave unused when using multiprocessing, for appropriate other applications' performance. Cap degree of parallelism: <#total CPU cores>-UNUSED_CPU_CORES Default: 2.
        # Plotting parameters
        "benchmark_curve": 1,               # Default: 1, choose one: 1 or 2 for the benchmark CSV (BaselineDataCartPole_run1.csv or BaselineDataCartPole_run2.csv).
        "benchmark_name": "Baseline",       # Benchmark name to show in the legend for the benchmark curve. Default: "Baseline".
        "plot_smoothing_window": [201, 351],  # Use multiple values to plot multiple curves. Default: [1, 51, 101, 201, 251, 301]. Set to [1] to skip smoothing.
        "curve_confidence_interval": 0.6,   # Curve shading confidence interval. Default: 0.95. Set to 0 to skip CI shading.
        "curve_shaded_area_opacity": 0.06,  # Opacity of the shaded area for confidence intervals. Default: 0.05 (5% opacity).
        "show_curve_plots": True,           # Show learning curve plot at the end of or during the training.
        "separate_algorithm_plots": False,   # If True, each algorithm gets its own set of plots (one per smoothing window). Each algo's plots are saved to disk and (if show_curve_plots) shown non-blocking as soon as that algo finishes executing, so faster algos surface their plots first. Default: False (one combined plot per smoothing window).
        "animation_plot": False,            # Show CartPole animation at the end.
        "use_existing_disk_data": True,     # Whether to use existing data (.xlsx files) from disk if exists.
        "use_saved_disk_networks_checkpoints": False,
        # Environment
        "max_train_episode_length": 500, #500        # Episode truncation step. Default: 500.
        "base_seed": 42,                    # Base seed for CartPole environment and agent initialization. Each repetition will use a different seed derived from this base seed (e.g., base_seed + repetition_index).
        # Agent
        "n_timesteps": 1e6, #1e6,              # Total number of training timesteps. Default: 1000000.
        "max_eval_episode_length": 500, #500        # Episode truncation step. Default: 500.
        "eval_interval": 250,
        "eval_with_env_episode_trials": True, # Default: True. Set to False to use the fast proxy full episode return value from training (last_episode_return) for evaluation instead of running separate greedy environment episode trials via agent.evaluate(). Note: setting to False will speed up training and plotting, but will not provide true evaluation curves. Setting to True will provide true evaluation curves but will significantly increase training time due to the need to run separate evaluation episodes at each eval_interval.
        "n_eval_episodes": 5,
        "title_parameters": {               # [plot label, show flag]
            "n_repetitions": [r"Reps: ", True],
            "curve_confidence_interval": [r"CCI: ", False],
        },
        "legend_parameters": {              # [plot label, show flag]
            "use_saved_disk_networks_checkpoints": [r"CHP: ", False],
            "max_train_episode_length": [r"L: ", False],
            "max_eval_episode_length": [r"EvL: ", False],
            "eval_with_env_episode_trials": [r"EvEnv: ", False],
            "n_eval_episodes": [r"EvEp: ", False],
        },
    }
    ################[           End Global Parameters            ]################

    ################[ Algorithm Hyperparameters & Configurations ]##############
    # Select which algorithms to include in the training and plotting using included_algo_learnings.
    # Set value to True to include, False to exclude.
    included_algo_learnings = {
        "DQN": False,
        "REINFORCE": False,
        "AC": False,
        "A2C": False,
        "PPO": True,
    }
    # Using DQN implementation from the previous assignment (existing in the assignment2_repo directory)
    # ------------- Algorithm: DQN hyperparameters (optimal) ----
    DQN_config = {
        "gamma": 0.99,
        "learning_rate": [1e-3],
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

    ###############[ Training hyperparameters ]##############
    # ------------- Algorithm Type: REINFORCE hyperparameters (optimal) ----
    REINFORCE_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [1e-3],            # actor learning rate(s) to sweep
        "actor_hidden_nn": [[128, 128]],   # list of NN architectures to sweep
        # -- Engineering tricks (sweepable lists; optimal defaults) --
        "entropy_coef": [0.01],
        "max_grad_norm": [0.5],
        "adam_eps": [1e-5],
        "anneal_lr": [True],
        "orthogonal_init": [True],
        "normalize_advantages": [True],
        "normalize_obs": [False],
        "legend_parameters": {          # [plot label, show flag]
            "entropy_coef": [r"H: ", False],
            "max_grad_norm": [r"gclip: ", False],
            "adam_eps": [r"AdamEps: ", False],
            "anneal_lr": [r"anneal: ", False],
            "orthogonal_init": [r"ortho: ", False],
            "normalize_advantages": [r"normA: ", False],
            "normalize_obs": [r"normObs: ", False],
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
        },
    }
    # ------------- End REINFORCE hyperparameters -----------

    # ------------- Algorithm Type: AC hyperparameters (optimal) ----
    AC_config = {
        "gamma": [0.99],
        "actor_lr": [1e-3],                   # 2D actor learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],         # 2D list of NN architectures to sweep
        "critic_lr": [1e-3],                  # critic learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]],      # critic NN architectures to sweep
        "TN_step": [10],                      # list of n-step returns to sweep (Target Network). Default: [10]. Set to [1] to skip n-step return trials.
        # -- Engineering tricks (sweepable lists; optimal defaults) --
        "entropy_coef": [0.01],
        "max_grad_norm": [0.5],
        "adam_eps": [1e-5],
        "anneal_lr": [True],
        "orthogonal_init": [True],
        "normalize_advantages": [True],
        "normalize_obs": [False],
        "value_loss_coef": [0.5],
        "use_advantage": [True],
        "legend_parameters": {                 # [plot label, show flag]
            "entropy_coef": [r"H: ", False],
            "max_grad_norm": [r"gclip: ", False],
            "adam_eps": [r"AdamEps: ", False],
            "anneal_lr": [r"anneal: ", False],
            "orthogonal_init": [r"ortho: ", False],
            "normalize_advantages": [r"normA: ", False],
            "normalize_obs": [r"normObs: ", False],
            "value_loss_coef": [r"cV: ", False],
            "use_advantage": [r"adv: ", False],
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
    A2C_config = {
        "gamma": [0.99],                # list of discount factors to sweep
        "actor_lr": [1e-4],          # policy learning rate(s) to sweep
        "actor_hidden_nn": [64, 64],        # list of NN architectures to sweep for policy network
        "critic_lr": [0.01],        # value function learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]],  # list of NN architectures to sweep for value function network
        "TN_step": [10],                 # list of n-step returns to sweep (Target Network). Default: [10]. Set to [1] to skip n-step return trials.
        # -- Engineering tricks (sweepable lists; optimal defaults) --
        "entropy_coef": [0.01],
        "max_grad_norm": [0.5],
        "adam_eps": [1e-5],
        "anneal_lr": [True],
        "orthogonal_init": [True],
        "normalize_advantages": [True],
        "normalize_obs": [False],
        "value_loss_coef": [0.5],
        "activation_name": ["tanh"],
        "use_gae": [True],
        "gae_lambda": [0.95],
        "legend_parameters": {          # [plot label, show flag]
            "entropy_coef": [r"H: ", False],
            "max_grad_norm": [r"gclip: ", False],
            "adam_eps": [r"AdamEps: ", False],
            "anneal_lr": [r"anneal: ", False],
            "orthogonal_init": [r"ortho: ", False],
            "normalize_advantages": [r"normA: ", False],
            "normalize_obs": [r"normObs: ", False],
            "value_loss_coef": [r"cV: ", False],
            "activation_name": [r"act: ", False],
            "use_gae": [r"GAE: ", False],
            "gae_lambda": [r"lambdaGAE: ", False],
            "gamma": [r"$\gamma$: ", True],
            "actor_lr": [r"Actor $\alpha$: ", True],
            "critic_lr": [r"Critic $\beta$: ", True],
            "actor_hidden_nn": [r"Actor NN: ", True],
            "critic_hidden_nn": [r"Critic NN: ", True],
            "TN_step": [r"TN Step: ", False],
        },
    }
    # ------------- End A2C hyperparameters -----------

    # ------------- Algorithm Type: PPO hyperparameters (basic PPO + GAE) ----
    # Proximal Policy Optimisation (PPO-clipped) - Schulman
    PPO_config = {
        "gamma": [0.99],                  # list of discount factors to sweep
        "actor_lr": [3e-4],               # actor learning rate(s) to sweep
        "actor_hidden_nn": [[64, 64]],    # actor NN architectures to sweep
        "critic_lr": [1e-3],              # 4e-3 # critic learning rate(s) to sweep
        "critic_hidden_nn": [[128, 128]], # 256  # critic NN architectures to sweep
        "gae_lambda": [0.95],     # 0.96 # GAE lambda parameter which controls the bias-variance trade-off of the Generalized Advantage Estimation (GAE). Default: 0.95. Set to 1.0 to disable GAE and use regular advantage estimation.
        "clip_eps": [0.2],                # 0.1  # PPO clipping epsilon which controls the clipping range for the probability ratio in the PPO surrogate objective. Default: 0.2.
        "n_epochs": [10,20,30],                 # 15   # of optimisation epochs per rollout which controls how many times we reuse each collected rollout batch of data to update the policy. Default: 10. Set to 1 to skip PPO epoch trials and only do one epoch per rollout.
        "rollout_steps": [2048],          # 1024 # of env steps per rollout (PPO buffer size) which controls how many steps of data we collect in each rollout before we perform policy updates. Default: 2048. Set to a large number (e.g., 1e6) to skip rollout length trials and effectively use the entire episode as one rollout.
        # -- Engineering tricks (sweepable lists; optimal defaults) --
        "entropy_coef": [0.01],
        "max_grad_norm": [0.5],
        "adam_eps": [1e-5],
        "anneal_lr": [True],
        "orthogonal_init": [True],
        "normalize_advantages": [True],
        "normalize_obs": [False],
        "value_loss_coef": [0.5],
        "activation_name": ["tanh"],
        "num_minibatches": [32],
        "clip_vloss": [True],
        "target_kl": [None],
        "legend_parameters": {            # [curve label, show flag]
            "entropy_coef": [r"H: ", False],
            "max_grad_norm": [r"gclip: ", False],
            "adam_eps": [r"AdamEps: ", False],
            "anneal_lr": [r"anneal: ", False],
            "orthogonal_init": [r"ortho: ", False],
            "normalize_advantages": [r"normA: ", False],
            "normalize_obs": [r"normObs: ", False],
            "value_loss_coef": [r"cV: ", False],
            "activation_name": [r"act: ", False],
            "num_minibatches": [r"mb: ", False],
            "clip_vloss": [r"vclip: ", False],
            "target_kl": [r"KL: ", False],
            "gamma": [r"$\gamma$: ", False],
            "actor_lr": [r"Actor $\alpha$: ", False],
            "critic_lr": [r"Critic $\beta$: ", False],
            "actor_hidden_nn": [r"Actor NN: ", False],
            "critic_hidden_nn": [r"Critic NN: ", False],
            "gae_lambda": [r"$\lambda_{GAE}$: ", False],
            "clip_eps": [r"$\epsilon_{clip}$: ", False],
            "n_epochs": [r"Epochs: ", True],
            "rollout_steps": [r"Rollout: ", False],
        },
    }
    # ------------- End PPO hyperparameters -----------

    ##########################################################

    ordered_algorithms = ["DQN", "REINFORCE", "AC", "A2C", "PPO"]
    experiments = [algo for algo in ordered_algorithms if included_algo_learnings.get(algo, False)]

    run_selected_experiments(
        experiments,
        global_config=global_config,
        REINFORCE_config=REINFORCE_config,
        AC_config=AC_config,
        A2C_config=A2C_config,
        DQN_config=DQN_config,
        PPO_config=PPO_config,
    )


if __name__ == '__main__':
    experiment()
