#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File / Excel utility functions for the DQN experiment pipeline.
Moved from assignment2_repo/Experiment.py.
"""

import os
import shutil
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from scipy.signal import savgol_filter
from scipy.stats import t as t_dist

from .DQN import run_dqn_trial_returns


def _fmt(v):
    """Format numbers safely for filenames."""
    if isinstance(v, float):
        return f"{v:.3g}".replace(".", "p")
    return str(v)


def _empty_data_sheets_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def _excel_cell_display_width(value):
    """Return the visible width of a value when exported to Excel."""
    if value is None:
        return 0
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return 0
    text = str(value)
    if not text:
        return 0
    return max(len(line) for line in text.splitlines())


def _autosize_excel_columns(worksheet, dataframe):
    """Resize Excel columns so their contents fit the widest cell in each column."""
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        max_width = _excel_cell_display_width(column_name)
        for value in dataframe[column_name].tolist():
            max_width = max(max_width, _excel_cell_display_width(value))
        worksheet.column_dimensions[get_column_letter(column_index)].width = max_width


def _save_results_to_excel(dir_path, base_filename, setting_jobs, setting_results):
    filepath = os.path.join(dir_path, f"{base_filename}.xlsx")
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for idx, (job, result) in enumerate(zip(setting_jobs, setting_results)):
            learning_curve, learning_curve_std, timesteps = result
            sheet_name = f"Setting_{idx + 1:03d}"
            data = {
                "timestep": timesteps,
                "learning_curve_mean": learning_curve,
                "learning_curve_std": learning_curve_std,
            }
            hp = job["hyperparams"]
            for key, value in hp.items():
                data[key] = value
            data["curve_label"] = job["curve_label"]
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            _autosize_excel_columns(writer.sheets[sheet_name], df)
    print(f"Saved {len(setting_results)} settings to {filepath}")


def _load_results_from_excel(filepath, n_settings):
    try:
        sheets = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
    except Exception as exc:
        raise ValueError(f"Failed to read Excel file '{filepath}': {exc}") from exc

    if len(sheets) != n_settings:
        raise ValueError(
            f"Sheet count mismatch: Excel has {len(sheets)} sheets but expected {n_settings}."
        )

    results = []
    required_columns = {"timestep", "learning_curve_mean", "learning_curve_std"}
    for sheet_name in sorted(sheets.keys()):
        df = sheets[sheet_name]
        missing_columns = sorted(required_columns.difference(set(df.columns)))
        if missing_columns:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing_columns)}"
            )
        try:
            timesteps = df["timestep"].values.astype(np.int32)
            learning_curve = df["learning_curve_mean"].values.astype(np.float32)
            learning_curve_std = df["learning_curve_std"].values.astype(np.float32)
        except Exception as exc:
            raise ValueError(
                f"Sheet '{sheet_name}' contains invalid numeric data: {exc}"
            ) from exc
        results.append((learning_curve, learning_curve_std, timesteps))

    return results


def smooth(y, window, poly=2):
    """
    y: vector to be smoothed
    window: size of the smoothing window
    """
    return savgol_filter(y, window, poly)


# ── Smoothing, benchmark, and repetition utilities ────────────────────────────

def _apply_optional_smoothing(learning_curve, plot_smoothing_window):
    if plot_smoothing_window is None:
        return learning_curve
    max_window = len(learning_curve) if len(learning_curve) % 2 == 1 else len(learning_curve) - 1
    window = min(int(plot_smoothing_window), max_window)
    if window >= 3:
        return smooth(learning_curve, window)
    return learning_curve


def _load_benchmark_curve(benchmark_curve, project_eval_interval, project_n_timesteps,
                          benchmark_eval_interval=250, episode_return_column="Episode_Return"):
    benchmark_files = {
        1: "BaselineDataCartPole_run1.csv",
        2: "BaselineDataCartPole_run2.csv",
    }
    if benchmark_curve not in benchmark_files:
        raise ValueError("benchmark_curve must be 1 or 2.")

    data = np.genfromtxt(benchmark_files[benchmark_curve], delimiter=",", names=True)
    if data.dtype.names is None or episode_return_column not in data.dtype.names:
        raise ValueError(
            f"Selected benchmark file does not contain requested column '{episode_return_column}'."
        )
    env_steps = np.atleast_1d(np.asarray(data["env_step"], dtype=np.float32))
    returns = np.atleast_1d(np.asarray(data[episode_return_column], dtype=np.float32))

    if env_steps.size == 0 or returns.size == 0:
        raise ValueError("Selected benchmark file has no usable data.")

    valid_rows = np.isfinite(env_steps) & np.isfinite(returns)
    env_steps = env_steps[valid_rows]
    returns = returns[valid_rows]

    if env_steps.size == 0:
        raise ValueError("Selected benchmark file contains only invalid rows.")

    sort_idx = np.argsort(env_steps)
    env_steps = env_steps[sort_idx]
    returns = returns[sort_idx]

    if project_eval_interval != benchmark_eval_interval and env_steps.size >= 2:
        normalized_steps = np.arange(
            env_steps[0],
            env_steps[-1] + project_eval_interval,
            project_eval_interval,
            dtype=np.float32,
        )
        normalized_steps = normalized_steps[normalized_steps <= env_steps[-1]]
        returns = np.interp(normalized_steps, env_steps, returns).astype(np.float32)
        env_steps = normalized_steps

    in_horizon = env_steps <= float(project_n_timesteps)
    env_steps = env_steps[in_horizon]
    returns = returns[in_horizon]

    if env_steps.size == 0:
        raise ValueError(
            "No benchmark points fall within project_n_timesteps. "
            "Increase n_timesteps or use a benchmark with earlier env_step values."
        )

    return env_steps, returns


def _run_single_repetition(**run_kwargs):
    return run_dqn_trial_returns(**run_kwargs)


def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy,
                    epsilon_start=None, epsilon_end=None, epsilon_decay=1.0, epsilon_decay_interval=1,
                    softmax_temp=None, plot_smoothing_window=None, plot=False, eval_interval=500,
                    nn_hidden_layer_widths=np.array([32, 32]), TN_step=10,
                    target_network_active=True,
                    base_seed=42, changing_hyperparameters_text=None,
                    trial_run_start=0, total_trial_runs=None,
                    parallel_executor=None, parallel_workers=1,
                    er_active=False, er_replay_buffer_size=10000,
                    er_batch_size=64, er_min_replay_size=100,
                    er_sample_train_frequency=1, er_replay_ratio=1.0,
                    enable_progress_bar=True, emit_trial_header=True):


    if backup != 'q':
        raise ValueError("This project supports only backup='q' (DQN).")
    if policy not in ('egreedy', 'softmax'):
        raise ValueError("policy must be 'egreedy' or 'softmax'.")

    returns_over_repetitions = []
    timesteps = None
    now = time.time()

    if policy == 'egreedy':
        if epsilon_start is None:
            raise ValueError("policy='egreedy' requires epsilon_start.")
        start_eps = float(epsilon_start)
        end_eps = start_eps if epsilon_end is None else float(epsilon_end)
        policy_kwargs = {
            "exploration_method": "epsilon_greedy",
            "epsilon_start": start_eps,
            "epsilon_end": end_eps,
            "epsilon_decay": float(epsilon_decay),
            "epsilon_decay_interval": epsilon_decay_interval,
            "softmax_temp": None,
        }
    else:
        if softmax_temp is None:
            raise ValueError("policy='softmax' requires softmax_temp.")
        policy_kwargs = {
            "exploration_method": "softmax",
            "epsilon_start": None,
            "epsilon_end": None,
            "epsilon_decay": None,
            "epsilon_decay_interval": 1,
            "softmax_temp": float(softmax_temp),
        }

    rep_outputs = [None] * n_repetitions
    use_parallel = parallel_workers is not None and parallel_workers > 1 and n_repetitions > 1

    if use_parallel:
        own_executor = parallel_executor is None
        executor: Executor | None = parallel_executor
        if own_executor:
            max_workers = min(int(parallel_workers), n_repetitions)
            executor = ProcessPoolExecutor(max_workers=max_workers)
        assert executor is not None
        try:
            future_to_rep = {}
            for rep in range(n_repetitions):
                run_seed = base_seed + rep
                trial_run_index = trial_run_start + rep + 1
                progress_bar_position = trial_run_start + rep if enable_progress_bar else None
                run_kwargs = dict(
                    n_env_steps=n_timesteps,
                    max_episode_length=max_episode_length,
                    learning_rate=learning_rate,
                    nn_hidden_layer_widths=nn_hidden_layer_widths,
                    discount_factor=gamma,
                    target_network_step=TN_step,
                    target_network_active=bool(target_network_active),
                    n_returns_interval=eval_interval,
                    seed=run_seed,
                    changing_hyperparameters_text=changing_hyperparameters_text,
                    trial_run_index=trial_run_index,
                    total_trial_runs=total_trial_runs,
                    enable_progress_bar=enable_progress_bar,
                    emit_trial_header=False,
                    progress_bar_position=progress_bar_position,
                    progress_bar_desc=f"Trial {trial_run_index}/{total_trial_runs}",
                    er_active=er_active,
                    er_replay_buffer_size=er_replay_buffer_size,
                    er_batch_size=er_batch_size,
                    er_min_replay_size=er_min_replay_size,
                    er_sample_train_frequency=er_sample_train_frequency,
                    er_replay_ratio=er_replay_ratio,
                    **policy_kwargs,
                )
                future = executor.submit(_run_single_repetition, **run_kwargs)
                future_to_rep[future] = rep

            for future in as_completed(future_to_rep):
                rep = future_to_rep[future]
                rep_outputs[rep] = future.result()
        finally:
            if own_executor and executor is not None:
                executor.shutdown()
    else:
        for rep in range(n_repetitions):
            run_seed = base_seed + rep
            trial_run_index = trial_run_start + rep + 1
            progress_bar_position = trial_run_start + rep if enable_progress_bar else None
            rep_outputs[rep] = _run_single_repetition(
                n_env_steps=n_timesteps,
                max_episode_length=max_episode_length,
                learning_rate=learning_rate,
                nn_hidden_layer_widths=nn_hidden_layer_widths,
                discount_factor=gamma,
                target_network_step=TN_step,
                target_network_active=bool(target_network_active),
                n_returns_interval=eval_interval,
                seed=run_seed,
                changing_hyperparameters_text=changing_hyperparameters_text,
                trial_run_index=trial_run_index,
                total_trial_runs=total_trial_runs,
                enable_progress_bar=enable_progress_bar,
                emit_trial_header=emit_trial_header,
                progress_bar_position=progress_bar_position,
                progress_bar_desc=f"Trial {trial_run_index}/{total_trial_runs}",
                er_active=er_active,
                er_replay_buffer_size=er_replay_buffer_size,
                er_batch_size=er_batch_size,
                er_min_replay_size=er_min_replay_size,
                er_sample_train_frequency=er_sample_train_frequency,
                er_replay_ratio=er_replay_ratio,
                **policy_kwargs,
            )

    for returns, rep_timesteps in rep_outputs:
        returns_over_repetitions.append(np.asarray(returns, dtype=np.float32))
        if timesteps is None:
            timesteps = np.asarray(rep_timesteps, dtype=np.int32)

    if emit_trial_header:
        print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    all_returns = np.array(returns_over_repetitions)
    learning_curve = np.mean(all_returns, axis=0)
    learning_curve_std = np.std(all_returns, axis=0, ddof=1) if all_returns.shape[0] > 1 else np.zeros_like(learning_curve)
    learning_curve = _apply_optional_smoothing(learning_curve, plot_smoothing_window)
    learning_curve_std = _apply_optional_smoothing(learning_curve_std, plot_smoothing_window)
    return learning_curve, learning_curve_std, timesteps
