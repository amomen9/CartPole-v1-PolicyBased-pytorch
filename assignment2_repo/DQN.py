import numpy as np
import random
import torch

from .mylibrary import (
    PolicyNetwork,
    choose_action,
    env,
    reset_dqn_target_cache,
    train_dqn,
)


def run_dqn_trial(
    n_env_steps,
    max_episode_length,
    learning_rate,
    nn_hidden_layer_widths,
    discount_factor,
    target_network_step,
    n_returns_interval,
    exploration_method,
    target_network_active=True,
    epsilon_start=None,
    epsilon_end=None,
    epsilon_decay=None,
    epsilon_decay_interval=1,
    softmax_temp=None,
    seed=None,
    changing_hyperparameters_text=None,
    trial_run_index=None,
    total_trial_runs=None,
    enable_progress_bar=True,
    emit_trial_header=True,
    progress_bar_position=None,
    progress_bar_desc="Env Steps",
    shared_step_counter=None,
    er_active=False,
    er_replay_buffer_size=10000,
    er_batch_size=64,
    er_min_replay_size=100,
    er_sample_train_frequency=1,
    er_replay_ratio=1.0,
    max_eval_episode_length=None,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
    full_episode_updates: bool = False,
):
    """Run one DQN training trial and return model plus learning-curve data."""
    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))

    nn_hidden_layer_widths = np.asarray(nn_hidden_layer_widths)
    nn_depth = len(nn_hidden_layer_widths) + 2

    model = PolicyNetwork(nn_depth=nn_depth, nn_hidden_layer_widths=nn_hidden_layer_widths)

    # Ensure target-network cache does not leak state between trials.
    reset_dqn_target_cache()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # if changing_hyperparameters_text:
    #     print(changing_hyperparameters_text, flush=True)
    if emit_trial_header and trial_run_index is not None and total_trial_runs is not None:
        print(f"Trial run: {trial_run_index}/{total_trial_runs}", flush=True)

    returns, timesteps = train_dqn(
        model,
        optimizer,
        env,
        n_env_steps=n_env_steps,
        max_episode_length=max_episode_length,
        max_eval_episode_length=max_eval_episode_length,
        eval_with_env_episode_trials=eval_with_env_episode_trials,
        n_eval_episodes=n_eval_episodes,
        discount_factor=discount_factor,
        target_network_step=target_network_step,
        target_network_active=bool(target_network_active),
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        epsilon_decay_interval=epsilon_decay_interval,
        n_returns_interval=n_returns_interval,
        exploration_method=exploration_method,
        softmax_temp=softmax_temp,
        enable_progress_bar=enable_progress_bar,
        progress_bar_position=progress_bar_position,
        progress_bar_desc=progress_bar_desc,
        er_active=er_active,
        er_replay_buffer_size=er_replay_buffer_size,
        er_batch_size=er_batch_size,
        er_min_replay_size=er_min_replay_size,
        er_sample_train_frequency=er_sample_train_frequency,
        er_replay_ratio=er_replay_ratio,
        shared_step_counter=shared_step_counter,
        full_episode_updates=full_episode_updates,
    )
    return model, returns, timesteps


def run_dqn_trial_returns(
    n_env_steps,
    max_episode_length,
    learning_rate,
    nn_hidden_layer_widths,
    discount_factor,
    target_network_step,
    n_returns_interval,
    exploration_method,
    target_network_active=True,
    epsilon_start=None,
    epsilon_end=None,
    epsilon_decay=None,
    epsilon_decay_interval=1,
    softmax_temp=None,
    seed=None,
    changing_hyperparameters_text=None,
    trial_run_index=None,
    total_trial_runs=None,
    enable_progress_bar=True,
    emit_trial_header=True,
    progress_bar_position=None,
    progress_bar_desc="Env Steps",
    shared_step_counter=None,
    er_active=False,
    er_replay_buffer_size=10000,
    er_batch_size=64,
    er_min_replay_size=100,
    er_sample_train_frequency=1,
    er_replay_ratio=1.0,
    max_eval_episode_length=None,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
    full_episode_updates: bool = False,
):
    """Run one DQN trial and return only arrays to reduce IPC overhead."""
    _, returns, timesteps = run_dqn_trial(
        n_env_steps=n_env_steps,
        max_episode_length=max_episode_length,
        max_eval_episode_length=max_eval_episode_length,
        eval_with_env_episode_trials=eval_with_env_episode_trials,
        n_eval_episodes=n_eval_episodes,
        learning_rate=learning_rate,
        nn_hidden_layer_widths=nn_hidden_layer_widths,
        discount_factor=discount_factor,
        target_network_step=target_network_step,
        target_network_active=bool(target_network_active),
        n_returns_interval=n_returns_interval,
        exploration_method=exploration_method,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        epsilon_decay_interval=epsilon_decay_interval,
        softmax_temp=softmax_temp,
        seed=seed,
        changing_hyperparameters_text=changing_hyperparameters_text,
        trial_run_index=trial_run_index,
        total_trial_runs=total_trial_runs,
        enable_progress_bar=enable_progress_bar,
        emit_trial_header=emit_trial_header,
        progress_bar_position=progress_bar_position,
        progress_bar_desc=progress_bar_desc,
        er_active=er_active,
        er_replay_buffer_size=er_replay_buffer_size,
        er_batch_size=er_batch_size,
        er_min_replay_size=er_min_replay_size,
        er_sample_train_frequency=er_sample_train_frequency,
        er_replay_ratio=er_replay_ratio,
        shared_step_counter=shared_step_counter,
        full_episode_updates=full_episode_updates,
    )
    return np.asarray(returns, dtype=np.float32), np.asarray(timesteps, dtype=np.int32)


def neural_net_policy(obs, model, exploration_method="epsilon_greedy", softmax_temp=1.0):
    """Select an action from a trained model for visualization/evaluation."""
    with torch.no_grad():
        if exploration_method == "softmax":
            return choose_action(model, obs, exploration_method="softmax", temp=softmax_temp)
        return choose_action(model, obs, exploration_method="epsilon_greedy", epsilon=0.0)
