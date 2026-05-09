#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import numpy as np
import time
from assignment2_repo.Helper import LearningCurvePlot, smooth
from assignment2_repo.functions import _apply_optional_smoothing, _load_benchmark_curve, average_over_repetitions
from .functions import _fmt, _empty_data_sheets_dir, _save_results_to_excel, _load_results_from_excel
from .DQN import run_dqn_trial_returns



def experiment():

    #################[ Parameters & Hyperparameters ]################
    benchmark_curve = 1         # Default: 1, choose one: 1 or 2 for the dataset that is given for the benchmark purpose in the assignment. See _load_benchmark_curve() for details.
    # Experiment
 
    n_repetitions = 3           # Default: 5
    # ------------- Plotting parameters --------------
    plot = False
    plot_smoothing_window = np.array([1,51,101,201,251,301])#251        # [0]: non-smoothed, [1]: smoothed, Default for smoothed: 9, Optimal for smoothed: 9. Set the smoothed window to 1 to skip smoothing (i.e. non-smoothed and smoothed curves will be identical).
    curve_confidence_interval = 0.6    # Demonstrated confidence interval for the shaded area around the learning curves. Default: 0.95, Optimal: 0.97. Set to 0 to skip confidence interval shading.
    curve_shaded_area_opacity = 0.05             # Opacity of the shaded confidence interval area around the learning curves. Default: 0.2
    use_existing_disk_data = True          # Whether to use existing data from disk if available, or to re-run all experiments from scratch. Default: True. Set to False to force re-running all experiments even if data files are present.
    
    # --------------- DQN environment ----------------
    n_timesteps = 100000#000        # Default: 10e6, Satisfactory results min (Usually roughly necessary and sufficient): 200000
    eval_interval = 250         # Default: 250
    max_episode_length = 500    # Default: 500
    # Back-up method
    backup = 'q'                # Only DQN is supported in this assignment, so backup='q' is fixed here.   

    # -------------- DQN hyperparameters --------------
    gamma = 0.99                # Default: 0.95, Optimal: 0.99
    learning_rate = np.array([0.001])       # Default: 0.05, Optimal: 0.001
    nn_hidden_layer_widths = np.array([128,128])  # Default: [np.array([64, 64])], Optimal: [np.array([128, 128])]
    nn_include_hp_in_legend = False                   # Whether to include neural network architecture and gamma in the legend text for each curve, in addition to the base hyperparameters. Default: True.
    nn_include_lr_in_legend = False                   # Whether to include learning rate in the legend text for each curve, in addition to the base hyperparameters. Default: True.

    
    # ----------- Exploration hyperparameters -------------
    # epsilon-decay values
    epsilon_start = 1                         # Default: 0.1, Optimal: 1.0
    epsilon_end = 0.02                          # Default: 0.0001, Optimal: 0.02    
    epsilon_decay = 0.9995                      # Default: 0.995, Optimal: 0.9995
    epsilon_decay_interval = 0                  # Default: 5, Optimal: 1 - Set to 0 to skip epsilon decay trials.
    epsilon_decay_enabled = epsilon_decay_interval > 0
    # Ordinary epsilon greedy
    epsilons = np.array([0.05])      # Default: np.array([epsilon_start, epsilon_start**2, epsilon_start**3]), Optimal: np.array([0.1, 0.05, 0.02])
    softmax_temps = np.array([1.0, 0.5, 0.1])   # Default: np.array([0.001, 0.01, 0.1]), Optimal: np.array([1.0, 0.5, 0.1])
    exploration_method = "egreedy"              # choose one: "egreedy" or "softmax"
    exp_include_hp_in_legend = False                   # Whether to include exploration hyperparameters in the legend text for each curve, in addition to the base hyperparameters. Default: True.
    
    # --------- Methods ------------
    # Activate/Deactivate methods for ablation study (Target Network and Experience Replay).
    # Target Network
    TN_step	= np.array([100])   		 # Interval for updating target network (Default: 10), Optimal: 100							 #
    TN_active = np.array([True])  # Default: np.array([True]), Optimal: np.array([False, True])
    TN_include_hp_in_legend = True      # Whether to include Target Network hyperparameters in the legend text for each curve, in addition to the base hyperparameters. Default: True.
    # Experience Replay
    ER_replay_buffer_size = 80000        # Default: 10000, Optimal: 80000
    ER_batch_size = np.array([64])                   # Default: 16, Optimal: 64  
    ER_min_replay_size = 2000            # Default: 100, Optimal: 2000
    ER_sample_train_frequency = np.array([1])        # Default: 20, Optimal: 1
    ER_replay_ratio = 1.0                # Default: 1.0, Optimal: 1.0
    ER_active = np.array([True])  # Default: np.array([True])
    ER_include_hp_in_legend = True   # Whether to include Experience Replay hyperparameters in the legend text for each curve, in addition to the base hyperparameters. Default: True.

    # ---------- PyTorch compilation/runtime ------------
    # torchscript_train_mode: choose one of "none", "script", "trace".
    # Keep trace off by default because this project mixes single-state and batch forward passes.
    compilation_config = {
        "enable_torch_compile": True,
        "torch_compile_backend": "inductor",
        "torch_compile_fallback_backend": None,
        "torch_compile_mode": "reduce-overhead",
        "torch_compile_dynamic": False,
        "torch_compile_fullgraph": False,
        "torch_compile_disable": False,
        "torch_compile_prewarm": True,
        "reuse_compiled_model_across_trials": True,
        "torchscript_train_mode": "none",
        "matmul_precision": "high",
        "enable_cuda_tf32": True,
    }
    ######################################################################

    benchmark_name = "Baseline"      # Default legend label for benchmark curve.

    start_time = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Experiment started at: {start_human}\n")
    with open("output_dqn.log", "w", encoding="utf-8") as f:
        f.write(f"Start the process at: {start_human}\n")
    
    base_seed = 42
        
    if exploration_method not in ("egreedy", "softmax"):
        raise ValueError("exploration_method must be either 'egreedy' or 'softmax'.")

    curve_confidence_interval = float(curve_confidence_interval)
    if curve_confidence_interval < 0.0 or curve_confidence_interval >= 1.0:
        raise ValueError("curve_confidence_interval must be in [0, 1). Use 0 to skip confidence interval shading.")
    shade_confidence_interval = curve_confidence_interval > 0.0
    curve_ci_alpha = 1.0 - curve_confidence_interval if shade_confidence_interval else None
    curve_shaded_area_opacity = float(curve_shaded_area_opacity)
    if curve_shaded_area_opacity < 0.0 or curve_shaded_area_opacity > 1.0:
        raise ValueError("curve_shaded_area_opacity must be in [0, 1].")

    plot_smoothing_windows = np.atleast_1d(np.asarray(plot_smoothing_window, dtype=np.int32))
    if plot_smoothing_windows.size < 1:
        raise ValueError("plot_smoothing_window must contain at least one smoothing window value.")
    if np.any(plot_smoothing_windows < 0):
        raise ValueError("plot_smoothing_window values must be non-negative integers.")

    epsilon_decay_enabled = bool(epsilon_decay_enabled)
    skip_decay_trials = exploration_method == "egreedy" and (not epsilon_decay_enabled)

    learning_rates = np.atleast_1d(np.asarray(learning_rate, dtype=np.float32))     # Learning rate array is normalized to 1D for easier handling of multiple values, even if only one value is given.
    if learning_rates.size == 0:
        raise ValueError("learning_rate must contain at least one value.")
    er_batch_sizes = np.atleast_1d(np.asarray(ER_batch_size, dtype=np.int32))
    if er_batch_sizes.size == 0:
        raise ValueError("ER_batch_size must contain at least one value.")
    if np.any(er_batch_sizes <= 0):
        raise ValueError("ER_batch_size values must be positive integers.")
    er_sample_train_frequencies = np.atleast_1d(np.asarray(ER_sample_train_frequency, dtype=np.int32))
    if er_sample_train_frequencies.size == 0:
        raise ValueError("ER_sample_train_frequency must contain at least one value.")
    if np.any(er_sample_train_frequencies <= 0):
        raise ValueError("ER_sample_train_frequency values must be positive integers.")
    architectures = _as_architecture_list(nn_hidden_layer_widths)

    n_learning_rates = int(len(learning_rates))
    n_target_steps = int(len(TN_step))
    n_architectures = int(len(architectures))
    if exploration_method == "egreedy":
        runs_per_combo = len(epsilons) + (0 if skip_decay_trials else 1)
    else:
        runs_per_combo = 1 + len(softmax_temps)
    n_tn_step_combos = sum(n_target_steps if bool(tn) else 1 for tn in TN_active)
    n_er_combos = len(er_batch_sizes) * len(er_sample_train_frequencies) * len(ER_active)
    total_trial_runs = n_repetitions * n_learning_rates * n_tn_step_combos * n_er_combos * n_architectures * runs_per_combo
    parallel_workers = max(1, min(total_trial_runs, os.cpu_count() or 1))
    trial_run_start = 0

    lr_txt = "-".join(_fmt(float(lr)) for lr in learning_rates)
    if exploration_method == "egreedy":
        eps_txt = "-".join(_fmt(float(e)) for e in epsilons)
        base_filename = (
            f"dqn_lr{lr_txt}"
            f"_eps{eps_txt}"
            f"_rep{n_repetitions}_ts{n_timesteps}_eval{eval_interval}"
        )
    else:
        temp_txt = "-".join(_fmt(float(t)) for t in softmax_temps)
        base_filename = (
            f"dqn_lr{lr_txt}"
            f"_temp{temp_txt}"
            f"_rep{n_repetitions}_ts{n_timesteps}_eval{eval_interval}"
        )

    optimal_episode_return = 500.0
    
    
    ####### Experiments (DQN only)
    benchmark_steps, benchmark_returns_raw = _load_benchmark_curve(
        benchmark_curve=benchmark_curve,
        project_eval_interval=eval_interval,
        project_n_timesteps=n_timesteps,
        episode_return_column="Episode_Return",
    )

    plot_configs = []
    for window in plot_smoothing_windows:
        window = int(window)
        is_not_smoothed = window <= 1
        benchmark_returns = _apply_optional_smoothing(
            np.asarray(benchmark_returns_raw, dtype=np.float32), window
        )
        benchmark_returns = np.minimum(benchmark_returns, float(optimal_episode_return))
        if is_not_smoothed:
            plot_obj = LearningCurvePlot(title="DQN - not smoothed plot")
        else:
            plot_obj = LearningCurvePlot(title="DQN - smoothed plot")

        plot_configs.append({
            "window": window,
            "is_not_smoothed": is_not_smoothed,
            "plot": plot_obj,
            "benchmark_steps": benchmark_steps,
            "benchmark_returns": benchmark_returns,
        })

    setting_jobs = []
    for tn_active in TN_active:
        tn_active_bool = bool(tn_active)
        effective_tn_steps = TN_step if tn_active_bool else np.array([1])
        for er_active_val in ER_active:
            er_active_bool = bool(er_active_val)
            effective_er_batch_sizes = er_batch_sizes
            for er_batch_size_value in effective_er_batch_sizes:
                er_batch_size_value = int(er_batch_size_value)
                effective_er_sample_train_frequencies = er_sample_train_frequencies
                for er_sample_train_frequency_value in effective_er_sample_train_frequencies:
                    er_sample_train_frequency_value = int(er_sample_train_frequency_value)
                    for learning_rate_value in learning_rates:
                        learning_rate_value = float(learning_rate_value)
                        for target_step in effective_tn_steps:
                            target_step = int(target_step)
                            for hidden_widths in architectures:
                                architecture_label = _nn_label(hidden_widths)
                                tn_tag = "on" if tn_active_bool else "off"
                                er_tag = "on" if er_active_bool else "off"
                                label_parts = []

                                if (not tn_active_bool) and (not er_active_bool):
                                    label_parts.append("NAIVE")
                                else:
                                    if tn_active_bool:
                                        if TN_include_hp_in_legend:
                                            label_parts.append(f"TN: {target_step}")
                                        else:
                                            label_parts.append("TN")
                                    elif TN_include_hp_in_legend:
                                        label_parts.append("TN: inactive")

                                    if er_active_bool:
                                        if ER_include_hp_in_legend:
                                            label_parts.append(
                                                "ER: buf={}, batch={}, min={}, freq={}, ratio={}".format(
                                                    int(ER_replay_buffer_size),
                                                    int(er_batch_size_value),
                                                    int(ER_min_replay_size),
                                                    int(er_sample_train_frequency_value),
                                                    _fmt(float(ER_replay_ratio)),
                                                )
                                            )
                                        else:
                                            label_parts.append("ER")
                                    elif ER_include_hp_in_legend:
                                        label_parts.append("ER: inactive")

                                if nn_include_hp_in_legend:
                                    label_parts.append(architecture_label)
                                    label_parts.append(f"gamma={_fmt(float(gamma))}")

                                if nn_include_lr_in_legend:
                                    label_parts.append(f"lr={_fmt(learning_rate_value)}")

                                run_label_prefix = ", ".join(label_parts)

                                er_kwargs = dict(
                                    er_active=er_active_bool,
                                    er_replay_buffer_size=ER_replay_buffer_size,
                                    er_batch_size=er_batch_size_value,
                                    er_min_replay_size=ER_min_replay_size,
                                    er_sample_train_frequency=er_sample_train_frequency_value,
                                    er_replay_ratio=ER_replay_ratio,
                                )

                                base_hyperparams = {
                                    "n_repetitions": n_repetitions,
                                    "n_timesteps": n_timesteps,
                                    "eval_interval": eval_interval,
                                    "max_episode_length": max_episode_length,
                                    "learning_rate": learning_rate_value,
                                    "gamma": gamma,
                                    "nn_hidden_layer_widths": str(hidden_widths.tolist()),
                                    "epsilon_decay_enabled": epsilon_decay_enabled,
                                    "epsilons": str(epsilons.tolist()),
                                    "softmax_temps": str(softmax_temps.tolist()),
                                    "exploration_method": exploration_method,
                                    "TN_active": tn_active_bool,
                                    "TN_step": target_step,
                                    "ER_active": er_active_bool,
                                    "ER_replay_buffer_size": int(ER_replay_buffer_size),
                                    "ER_batch_size": int(er_batch_size_value),
                                    "ER_min_replay_size": int(ER_min_replay_size),
                                    "ER_sample_train_frequency": int(er_sample_train_frequency_value),
                                    "ER_replay_ratio": float(ER_replay_ratio),
                                    "compilation_config": json.dumps(compilation_config, sort_keys=True),
                                }

                                common_kwargs = dict(
                                    backup=backup,
                                    n_repetitions=n_repetitions,
                                    n_timesteps=n_timesteps,
                                    max_episode_length=max_episode_length,
                                    learning_rate=learning_rate_value,
                                    gamma=gamma,
                                    policy=exploration_method,
                                    plot_smoothing_window=1,
                                    plot=plot,
                                    eval_interval=eval_interval,
                                    nn_hidden_layer_widths=hidden_widths,
                                    TN_step=target_step,
                                    target_network_active=tn_active_bool,
                                    base_seed=base_seed,
                                    trial_run_start=trial_run_start,
                                    total_trial_runs=total_trial_runs,
                                    compilation_config=compilation_config,
                                    **er_kwargs,
                                )

                                if exploration_method == "egreedy":
                                    base_change_text = _format_changing_hyperparameters([
                                        ("learning_rate", learning_rate_value),
                                        ("TN_active", tn_tag),
                                        ("TN_step", target_step),
                                        ("ER_active", er_tag),
                                        ("ER_batch_size", er_batch_size_value),
                                        ("ER_sample_train_frequency", er_sample_train_frequency_value),
                                        ("nn_hidden_layer_widths", np.asarray(hidden_widths, dtype=np.int32)),
                                        ("epsilon_start", float(epsilon_start)),
                                        ("epsilon_end", float(epsilon_end)),
                                        ("epsilon_decay", float(epsilon_decay)),
                                        ("epsilon_decay_interval", int(epsilon_decay_interval)),
                                    ])
                                else:
                                    base_change_text = _format_changing_hyperparameters([
                                        ("learning_rate", learning_rate_value),
                                        ("TN_active", tn_tag),
                                        ("TN_step", target_step),
                                        ("ER_active", er_tag),
                                        ("ER_batch_size", er_batch_size_value),
                                        ("ER_sample_train_frequency", er_sample_train_frequency_value),
                                        ("nn_hidden_layer_widths", np.asarray(hidden_widths, dtype=np.int32)),
                                        ("softmax_temp", float(softmax_temps[0])),
                                    ])

                                if not skip_decay_trials:
                                    if exploration_method == "egreedy":
                                        if exp_include_hp_in_legend:
                                            exploration_label = r'$\epsilon$-greedy decay, $\epsilon$: {}$\rightarrow${}'.format(
                                                _fmt(epsilon_start), _fmt(epsilon_end)
                                            )
                                        else:
                                            exploration_label = None
                                    else:
                                        if exp_include_hp_in_legend:
                                            exploration_label = r'softmax, $\tau$ = {}'.format(
                                                _fmt(float(softmax_temps[0]))
                                            )
                                        else:
                                            exploration_label = 'softmax'
                                    curve_label = _compose_curve_label(run_label_prefix, exploration_label)
                                    setting_jobs.append({
                                        "curve_label": curve_label,
                                        "kwargs": {
                                            **common_kwargs,
                                            "epsilon_start": epsilon_start,
                                            "epsilon_end": epsilon_end,
                                            "epsilon_decay": epsilon_decay,
                                            "epsilon_decay_interval": epsilon_decay_interval,
                                            "softmax_temp": float(softmax_temps[0]) if exploration_method == "softmax" else None,
                                            "changing_hyperparameters_text": base_change_text,
                                        },
                                        "hyperparams": {**base_hyperparams,
                                            "epsilon_start": epsilon_start if exploration_method == "egreedy" else None,
                                            "epsilon_end": epsilon_end if exploration_method == "egreedy" else None,
                                            "epsilon_decay": epsilon_decay if exploration_method == "egreedy" else None,
                                            "epsilon_decay_interval": epsilon_decay_interval if exploration_method == "egreedy" else None,
                                        },
                                    })
                                    trial_run_start += n_repetitions

                                if exploration_method == "egreedy":
                                    for epsilon in epsilons:
                                        loop_change_text = _format_changing_hyperparameters([
                                            ("learning_rate", learning_rate_value),
                                            ("TN_active", tn_tag),
                                            ("TN_step", target_step),
                                            ("ER_active", er_tag),
                                            ("ER_batch_size", er_batch_size_value),
                                            ("ER_sample_train_frequency", er_sample_train_frequency_value),
                                            ("nn_hidden_layer_widths", np.asarray(hidden_widths, dtype=np.int32)),
                                            ("epsilon_start", float(epsilon)),
                                            ("epsilon_end", float(epsilon)),
                                            ("epsilon_decay", 1.0),
                                            ("epsilon_decay_interval", 1),
                                        ])
                                        if exp_include_hp_in_legend:
                                            exploration_label = r'$\epsilon$-greedy, $\epsilon$ = {}'.format(
                                                _fmt(float(epsilon))
                                            )
                                        else:
                                            exploration_label = None
                                        curve_label = _compose_curve_label(run_label_prefix, exploration_label)
                                        setting_jobs.append({
                                            "curve_label": curve_label,
                                            "kwargs": {
                                                **common_kwargs,
                                                "epsilon_start": float(epsilon),
                                                "epsilon_end": float(epsilon),
                                                "epsilon_decay": 1.0,
                                                "epsilon_decay_interval": 1,
                                                "softmax_temp": None,
                                                "changing_hyperparameters_text": loop_change_text,
                                            },
                                            "hyperparams": {**base_hyperparams,
                                                "epsilon_start": float(epsilon),
                                                "epsilon_end": float(epsilon),
                                                "epsilon_decay": 1.0,
                                                "epsilon_decay_interval": 1,
                                            },
                                        })
                                        trial_run_start += n_repetitions
                                else:
                                    for softmax_temp in softmax_temps:
                                        loop_change_text = _format_changing_hyperparameters([
                                            ("learning_rate", learning_rate_value),
                                            ("TN_active", tn_tag),
                                            ("TN_step", target_step),
                                            ("ER_active", er_tag),
                                            ("ER_batch_size", er_batch_size_value),
                                            ("ER_sample_train_frequency", er_sample_train_frequency_value),
                                            ("nn_hidden_layer_widths", np.asarray(hidden_widths, dtype=np.int32)),
                                            ("softmax_temp", float(softmax_temp)),
                                        ])
                                        if exp_include_hp_in_legend:
                                            exploration_label = r'softmax, $\tau$ = {}'.format(
                                                _fmt(float(softmax_temp))
                                            )
                                        else:
                                            exploration_label = 'softmax'
                                        curve_label = _compose_curve_label(run_label_prefix, exploration_label)
                                        setting_jobs.append({
                                            "curve_label": curve_label,
                                            "kwargs": {
                                                **common_kwargs,
                                                "epsilon_start": None,
                                                "epsilon_end": None,
                                                "epsilon_decay": 1.0,
                                                "epsilon_decay_interval": 1,
                                                "softmax_temp": float(softmax_temp),
                                                "changing_hyperparameters_text": loop_change_text,
                                            },
                                            "hyperparams": {**base_hyperparams,
                                                "epsilon_start": None,
                                                "epsilon_end": None,
                                                "epsilon_decay": None,
                                                "epsilon_decay_interval": None,
                                            },
                                        })
                                        trial_run_start += n_repetitions

    # Allocate top headroom as a function of legend size so larger legend sets remain visible.
    expected_legend_entries = len(setting_jobs) + 2  # Curves + benchmark + CartPole optimum.
    legend_headroom = max(90, 80 + 20 * expected_legend_entries)
    for plot_config in plot_configs:
        plot_config["plot"].set_ylim(0, max_episode_length + legend_headroom)

    # --- Load existing results or run experiments ---
    data_sheets_dir = "data sheets"
    excel_path = os.path.join(data_sheets_dir, f"{base_filename}.xlsx")

    loaded_from_disk = False
    if use_existing_disk_data:
        if os.path.isfile(excel_path):
            try:
                setting_results = _load_results_from_excel(excel_path, len(setting_jobs))
            except Exception as exc:
                print(
                    "Existing Excel data is incompatible with current settings. "
                    f"Re-running experiments from scratch. Reason: {exc}"
                )
                loaded_from_disk = False
            else:
                loaded_from_disk = True
                print(f"Loaded {len(setting_results)} settings from existing data: {excel_path}")
        else:
            print(
                "No existing Excel data found in data sheets directory. "
                "Running experiments from scratch."
            )

    if not loaded_from_disk:
        setting_results = [None] * len(setting_jobs)
    parallelize_settings = parallel_workers > 1 and len(setting_jobs) > 1

    if not loaded_from_disk and parallelize_settings:
        setting_workers = min(parallel_workers, len(setting_jobs))
        with ProcessPoolExecutor(max_workers=setting_workers) as setting_executor:
            future_to_idx = {}
            for idx, job in enumerate(setting_jobs):
                run_kwargs = dict(job["kwargs"])
                run_kwargs["parallel_executor"] = None
                run_kwargs["parallel_workers"] = 1
                run_kwargs["emit_trial_header"] = False
                future = setting_executor.submit(average_over_repetitions, **run_kwargs)
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                setting_results[idx] = future.result()
    elif not loaded_from_disk:
        shared_executor = ProcessPoolExecutor(max_workers=parallel_workers) if parallel_workers > 1 else None
        try:
            for idx, job in enumerate(setting_jobs):
                run_kwargs = dict(job["kwargs"])
                run_kwargs["parallel_executor"] = shared_executor
                run_kwargs["parallel_workers"] = parallel_workers
                setting_results[idx] = average_over_repetitions(**run_kwargs)
        finally:
            if shared_executor is not None:
                shared_executor.shutdown()

    if not loaded_from_disk:
        _empty_data_sheets_dir(data_sheets_dir)
        _save_results_to_excel(data_sheets_dir, base_filename, setting_jobs, setting_results)

    for idx, job in enumerate(setting_jobs):
        learning_curve_raw, learning_curve_std_raw, timesteps = setting_results[idx]
        curve_label = job["curve_label"]
        for plot_config in plot_configs:
            window = int(plot_config["window"])
            learning_curve_window = _apply_optional_smoothing(
                np.asarray(learning_curve_raw, dtype=np.float32), window
            )
            learning_curve_std_window = _apply_optional_smoothing(
                np.asarray(learning_curve_std_raw, dtype=np.float32), window
            )

            y_upper_cap = float(optimal_episode_return)
            learning_curve_window = np.minimum(learning_curve_window, y_upper_cap)

            plot_obj = plot_config["plot"]
            plot_obj.add_curve(
                timesteps,
                learning_curve_window,
                label=curve_label,
            )
            if shade_confidence_interval:
                plot_obj.add_shaded_ci(
                    timesteps, learning_curve_window,
                    learning_curve_std_window, n=n_repetitions,
                    alpha=curve_ci_alpha,
                    fill_opacity=curve_shaded_area_opacity,
                    y_upper_cap=y_upper_cap,
                )

    for plot_config in plot_configs:
        plot_obj = plot_config["plot"]
        plot_obj.ax.plot(
            plot_config["benchmark_steps"],
            plot_config["benchmark_returns"],
            label=benchmark_name,
            ls=":",
            c="gray",
        )
        plot_obj.add_hline(optimal_episode_return, label="CartPole optimum")

    for plot_config in plot_configs:
        window = int(plot_config["window"])
        if window <= 1:
            suffix = f"w{window}-not-smoothed"
        else:
            suffix = f"w{window}-smoothed"
        plot_config["plot"].save(f"{base_filename}_{suffix}.png")

    total_execution_time = (time.perf_counter() - start_time) / 60.0
    with open("output_dqn.log", "a", encoding="utf-8") as f:
        f.writelines(f"Total execution time: {total_execution_time:.3f} minutes" + "\n")
    
    print(f"\nExperiment finished in {total_execution_time:.3f} minutes.")



def _nn_label(hidden_widths):
    widths = np.asarray(hidden_widths, dtype=np.int32)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("Each nn_hidden_layer_widths element must be a 1D non-empty sequence.")
    return "nn: <4,{},2>".format(",".join(str(int(w)) for w in widths))


def _as_architecture_list(nn_hidden_layer_widths):
    arr = np.asarray(nn_hidden_layer_widths, dtype=object)
    if arr.ndim == 1 and all(np.isscalar(v) for v in arr):
        return [np.asarray(arr, dtype=np.int32)]

    architectures = []
    for widths in nn_hidden_layer_widths:
        width_arr = np.asarray(widths, dtype=np.int32).reshape(-1)
        if width_arr.size == 0:
            raise ValueError("Each nn_hidden_layer_widths element must contain at least one width.")
        architectures.append(width_arr)

    if len(architectures) == 0:
        raise ValueError("nn_hidden_layer_widths must contain at least one architecture.")
    return architectures


def _format_changing_hyperparameters(hparams):
    parts = []
    for key, value in hparams:
        if isinstance(value, np.ndarray):
            value_text = "[{}]".format(",".join(str(int(v)) for v in value.astype(np.int32)))
        elif isinstance(value, float):
            value_text = str(value)
        else:
            value_text = str(value)
        parts.append(f"{key}:{value_text}")
    return "Changing hyperparameters: " + " ".join(parts)


def _compose_curve_label(prefix, exploration_text):
    if prefix and exploration_text:
        return f"{prefix}, {exploration_text}"
    if prefix:
        return prefix
    if exploration_text:
        return exploration_text
    return None


###########################################################################
# Main execution
if __name__ == '__main__':
    experiment()
