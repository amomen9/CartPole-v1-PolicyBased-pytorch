import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as visplt
import matplotlib.animation
import torch
import torch.nn as nn
import numpy as np
import Environment as environ
import functions as fn
import os
import time
from datetime import datetime
from Helper import (
    LearningCurvePlot,
    smooth,
)




# Begin Class CartPoleAgentPlot ##############################################################
class CartPoleAgentPlot:
    ''' Class for plotting CartPole agent behavior during training '''
    def __init__(self, env, title=None, curve_plot=False, animation_plot=False):
        self.env = env
        if curve_plot:
            self.fig,self.ax = visplt.subplots()
            ###### Setting plot parameters for better visualization
            self.ax.set_xlim(-2.4,2.4)
            self.ax.set_ylim(-0.5,0.5)
            self.ax.set_xlabel('Cart Position')
            self.ax.set_ylabel('Pole Angle (radians)')      
            visplt.rcParams['figure.figsize'] = (8,6) # set default figure size
            visplt.rcParams['animation.embed_limit'] = (1000*1000)*10*10  # 100 MB
            visplt.rcParamsDefault.update(visplt.rcParams) 
            visplt.rc('font', size=14)
            visplt.rc('axes', labelsize=14, titlesize=14)
            visplt.rc('legend', fontsize=14)
            visplt.rc('xtick', labelsize=10)
            visplt.rc('ytick', labelsize=10)
            visplt.rc('animation', html='jshtml')
        ##################################################################################################################
            if title is not None:
                self.ax.set_title(title)
    
    def add_episode(self,obs_history,label=None, ls="solid"):
        ''' obs_history: list of observations during episode (list of 4D vectors) '''
        cart_positions = [obs[0] for obs in obs_history]
        pole_angles = [obs[2] for obs in obs_history]
        if label is not None:
            self.ax.plot(cart_positions,pole_angles,label=label, ls=ls)
        else:
            self.ax.plot(cart_positions,pole_angles, ls=ls)


    def plot_environment(self, env, figsize=(5, 4)):
        visplt.figure(figsize=figsize)
        img = env.render()
        visplt.imshow(img)
        visplt.axis("off")
        return img

    # extra code – this cell displays an animation of one episode

    def update_scene(self, num, frames, patch):
        patch.set_data(frames[num])
        return patch,

    def plot_animation(self, frames, repeat=False, interval=40):
        fig = visplt.figure()
        patch = visplt.imshow(frames[0])
        visplt.axis('off')
        visplt.tight_layout(pad=0)

        # Use interactive-mode frame stepping instead of FuncAnimation so that
        # every single frame is guaranteed to be drawn on all backends (TkAgg
        # on Windows drops frames when timer-driven redraws can't keep up).
        pause_sec = interval / 1000.0
        visplt.ion()
        try:
            while True:
                for frame in frames:
                    if not visplt.fignum_exists(fig.number):
                        print(f"\nNumber of frames: {len(frames)}")
                        return fig
                    patch.set_data(frame)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    visplt.pause(pause_sec)
                if not repeat:
                    break
        finally:
            visplt.ioff()

        self._last_anim = fig          # keep figure alive
        print(f"Number of frames: {len(frames)}")
        return fig



    @staticmethod
    def _resolve_action(policy, obs):
        # Support passing a torch.nn.Module directly as policy.
        if isinstance(policy, nn.Module):
            with torch.no_grad():
                state = torch.as_tensor(obs, dtype=torch.float32)
                logit = policy(state)
                prob = torch.sigmoid(logit).item()
            return int(prob >= 0.5)

        action = policy(obs)
        if isinstance(action, tuple):
            action = action[0]
        if torch.is_tensor(action):
            action = action.item()
        return int(action)

    def test_one_episode(self, env, policy=None): # Test one episode of the environment with the given policy, and return the animation and number of steps.
        if policy is None:
            policy = self.test_policy
        if isinstance(policy, str):
            policy = getattr(self, policy, None) or globals().get(policy)
        if not callable(policy):
            raise TypeError("policy must be a callable or the name of a callable method")

        frames = []
        obs = env.obs
        while True:
            frames.append(env.render())
            action = self._resolve_action(policy, obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        env.close()
        return self.plot_animation(frames), len(frames)


# End Class CartPoleAgentPlot ##############################################################

# Example test policy: move left if pole angle is negative, right if pole angle is positive (this is a very simple heuristic that can solve the CartPole environment to some extent, but is not optimal).
@staticmethod
def test_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1
# End test policy definition ##############################################################

################[ Value_NN (Critic) ]################
class Value_NN(nn.Module):      # This is neural network for the critic (Values NN), which can be used as either a state-value critic V_phi(s) or an action-value critic Q_phi(s) depending on the output size and how it's trained.
    """State-value critic V_phi(s), or Q-value critic Q_phi(s) when used per-action."""
    def __init__(self, nn_hidden_layer_widths=np.array([64, 64]), output_size=1):
        super().__init__()
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_size = 4  # CartPole state dimension
        for width in hidden_widths:
            layers.append(nn.Linear(input_size, int(width)))
            layers.append(nn.ReLU())
            input_size = int(width)
        layers.append(nn.Linear(input_size, output_size))   # Output layer: 1 for V_phi(s), 2 for Q_phi(s) with two actions
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
#########################################################

################[ Policy_NN Class              ]################
class Policy_NN(nn.Module):
    def __init__(self, nn_hidden_layer_widths=np.array([5])):     # nn_depth is the total number of layers (input + hidden + output), and nn_hidden_layer_widths are the input and output sizes of all layers in depth order (the first number is for the input and the last number is for the output). For example, if nn_depth=3 and nn_hidden_layer_widths=[5, 5], then the network will have an input layer of size 4 (the state dimension), a hidden layer of size 5, another hidden layer of size 5, and an output layer of size 1 (the action dimension).
        super().__init__()        
        hidden_widths = np.asarray(nn_hidden_layer_widths, dtype=np.int32).tolist()
        if len(hidden_widths) == 0:
            raise ValueError("nn_hidden_layer_widths must contain at least one hidden-layer width")

        layers = []
        input_layer_size = 4  # input layer size (state dimension). Is always 4 for CartPole, but we keep it here for generality and readability.
        for width in hidden_widths:
            layers.append(nn.Linear(input_layer_size, int(width)))
            layers.append(nn.ReLU())
            input_layer_size = int(width)
        layers.append(nn.Linear(input_layer_size, 1))    # 1 is the output layer size (action dimension). Is always 1 for CartPole REINFORCE algorithm, but we keep it here for generality and readability.
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
####################################################################




# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_selected_experiments(
    experiments,
    *,
    global_config=None,
    reinforce_config=None,
    ac_config=None,
    a2c_config=None,
    dqn_config=None,
):
    """Orchestrate training, data loading, and plotting for all selected experiments.

    Parameters
    ----------
    experiments : list[str]
        Algorithm names to run. Supported: "REINFORCE", "AC", "A2C", "DQN".
    global_config : dict
        Global/shared parameters (benchmark, plotting, environment, seed).
    reinforce_config : dict or None
        REINFORCE-specific hyperparameters. Required when "REINFORCE" in experiments.
    ac_config : dict or None
        AC-specific hyperparameters. Required when "AC" in experiments.
    a2c_config : dict or None
        A2C-specific hyperparameters. Required when "A2C" in experiments.
    dqn_config : dict or None
        DQN-specific hyperparameters. Required when "DQN" in experiments.
    """
    # ── Unpack global config ──
    gc = global_config or {}
    benchmark_curve = gc.get("benchmark_curve", 1)
    benchmark_name = gc.get("benchmark_name", "Baseline")
    n_repetitions = int(gc.get("n_repetitions", 5))
    plot_smoothing_window = gc.get("plot_smoothing_window", np.array([1]))
    curve_confidence_interval = gc.get("curve_confidence_interval", 0.6)
    curve_shaded_area_opacity = gc.get("curve_shaded_area_opacity", 0.05)
    use_existing_disk_data = gc.get("use_existing_disk_data", True)
    format_sheets = bool(gc.get("format_sheets", False))
    formatted_sheets = bool(gc.get("formatted_sheets", False))
    n_timesteps = int(gc.get("n_timesteps", 100000))
    eval_interval = int(gc.get("eval_interval", 250))

    # Separate episode truncation lengths for training vs evaluation.
    # - max_train_episode_length is used to create the *training* environment (max_episode_steps).
    # - max_eval_episode_length is used when creating the *evaluation* environment inside agent.evaluate().
    default_episode_length = int(gc.get("max_episode_length", 500))
    max_train_episode_length = int(gc.get("max_train_episode_length", default_episode_length))
    max_eval_episode_length = int(gc.get("max_eval_episode_length", max_train_episode_length))

    # Evaluation mode:
    # - default False keeps the fast proxy from training (last_episode_return / previous_episode_return)
    # - True uses greedy environment episode trials via agent.evaluate()
    eval_with_env_episode_trials = bool(gc.get("eval_with_env_episode_trials", False))
    n_eval_episodes = int(gc.get("n_eval_episodes", 5))

    # If we evaluate using environment episode trials, the maximum achievable return is
    # the evaluation truncation length. Otherwise we use training truncation length.
    y_axis_episode_length_cap = max(max_train_episode_length, max_eval_episode_length)
    optimal_episode_return = float(
        max_eval_episode_length if eval_with_env_episode_trials else max_train_episode_length
    )

    base_seed = int(gc.get("base_seed", 42))
    curve_plot = gc.get("curve_plot", False)
    animation_plot = gc.get("animation_plot", False)

    start_time = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Experiment started at: {start_human}\n")
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Start the process at: {start_human}\n")
    print(f"Included experiments: {', '.join(experiments)}\n")

    from facilitation_functions import (
        _build_a2c_jobs,
        _build_ac_jobs,
        _build_algo_filename,
        _build_dqn_jobs,
        _build_reinforce_jobs,
        _load_all_excel_curves,
        _load_results_from_excel,
        _run_pending_parallel,
        save_algorithm_workbook,
    )

    # ── Validate parameters ──
    curve_confidence_interval = float(curve_confidence_interval)
    if curve_confidence_interval < 0.0 or curve_confidence_interval >= 1.0:
        raise ValueError("curve_confidence_interval must be in [0, 1).")
    shade_ci = curve_confidence_interval > 0.0
    curve_ci_alpha = 1.0 - curve_confidence_interval if shade_ci else None
    curve_shaded_area_opacity = float(curve_shaded_area_opacity)

    plot_smoothing_windows = np.atleast_1d(np.asarray(plot_smoothing_window, dtype=np.int32))
    if plot_smoothing_windows.size < 1:
        raise ValueError("plot_smoothing_window must contain at least one value.")

    # ── Build setting jobs for each selected algorithm (grouped by algo) ──
    algo_jobs = {}  # algo_upper -> list of setting_jobs
    algo_configs_map = {
        "REINFORCE": reinforce_config,
        "AC": ac_config,
        "A2C": a2c_config,
        "DQN": dqn_config,
    }
    # Disk-matching should depend on evaluation mode so proxy-vs-env-trial
    # results don't mix between runs.
    for _algo_key, _cfg in list(algo_configs_map.items()):
        if _cfg is not None:
            algo_configs_map[_algo_key] = {
                **_cfg,
                "max_train_episode_length": max_train_episode_length,
                "max_eval_episode_length": max_eval_episode_length,
                # Legacy key used by the DQN workbook hyperparams.
                "max_episode_length": max_train_episode_length,
                "eval_with_env_episode_trials": eval_with_env_episode_trials,
                "n_eval_episodes": n_eval_episodes,
            }
    all_setting_jobs = []
    for algo in experiments:
        algo_upper = algo.upper()
        jobs = []
        if algo_upper == "REINFORCE":
            if reinforce_config is None:
                raise ValueError("reinforce_config dict is required when REINFORCE is included.")
            jobs = _build_reinforce_jobs(
                algo_config=reinforce_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
            )
        elif algo_upper == "AC":
            if ac_config is None:
                raise ValueError("ac_config dict is required when AC is included.")
            jobs = _build_ac_jobs(
                algo_config=ac_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
            )
        elif algo_upper == "A2C":
            if a2c_config is None:
                raise ValueError("a2c_config dict is required when A2C is included.")
            jobs = _build_a2c_jobs(
                algo_config=a2c_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
            )
        elif algo_upper == "DQN":
            if dqn_config is None:
                raise ValueError("dqn_config dict is required when DQN is included.")
            jobs = _build_dqn_jobs(
                dqn_config=dqn_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        algo_jobs[algo_upper] = jobs
        all_setting_jobs.extend(jobs)

    # ── Load benchmark ──
    benchmark_steps, benchmark_returns_raw = _load_benchmark_curve(
        benchmark_curve=benchmark_curve,
        project_eval_interval=eval_interval,
        project_n_timesteps=n_timesteps,
        episode_return_column="Episode_Return",
    )

    # ── Prepare one plot per smoothing window ──
    title_tag = " + ".join(experiments)
    plot_configs = []
    for window in plot_smoothing_windows:
        window = int(window)
        is_not_smoothed = window <= 1
        benchmark_returns = _apply_optional_smoothing(
            np.asarray(benchmark_returns_raw, dtype=np.float32), window
        )
        benchmark_returns = np.minimum(benchmark_returns, float(optimal_episode_return))
        if is_not_smoothed:
            plot_obj = LearningCurvePlot(title=f"{title_tag} - not smoothed plot")
        else:
            plot_obj = LearningCurvePlot(title=f"{title_tag} - smoothed plot")
        plot_configs.append({
            "window": window,
            "is_not_smoothed": is_not_smoothed,
            "plot": plot_obj,
            "benchmark_steps": benchmark_steps,
            "benchmark_returns": benchmark_returns,
        })

    # ── Allocate top headroom for legend ──
    expected_legend_entries = len(all_setting_jobs) + 2
    legend_headroom = max(90, 80 + 20 * expected_legend_entries) * y_axis_episode_length_cap / 500.0
    for pc in plot_configs:
        pc["plot"].set_ylim(0, y_axis_episode_length_cap + legend_headroom)

    # ── Load existing results or run experiments (per algorithm) ──
    data_sheets_dir = "data sheets"
    os.makedirs(data_sheets_dir, exist_ok=True)

    setting_results = [None] * len(all_setting_jobs)
    algo_filenames = {}    # algo_upper -> base_filename
    algo_job_offsets = {}  # algo_upper -> start index in all_setting_jobs
    pending_settings = []  # (global_idx, job) - need to be computed
    algos_needing_save = set()

    # ── Pass 1: load from disk, collect and logically append what still needs running ──
    offset = 0
    for algo_upper, jobs in algo_jobs.items():
        algo_job_offsets[algo_upper] = offset
        n_jobs = len(jobs)
        cfg = algo_configs_map[algo_upper]
        base_filename = _build_algo_filename(algo_upper, cfg, n_repetitions, n_timesteps, eval_interval)
        algo_filenames[algo_upper] = base_filename
        excel_path = os.path.join(data_sheets_dir, f"{base_filename}.xlsx")

        if use_existing_disk_data and os.path.isfile(excel_path):
            try:
                algo_results, _mismatches = _load_results_from_excel(
                    excel_path,
                    cfg,
                    formatted_sheets=formatted_sheets,
                )
            except Exception as exc:
                print(f"[{algo_upper}] Existing Excel data is incompatible. Re-running from scratch. Reason: {exc}")
                algo_results = []
                _mismatches = {}

            if algo_results:
                print(f"[{algo_upper}] Loaded {len(algo_results)} matching setting(s) from: {excel_path}")
                loaded_count = min(len(algo_results), n_jobs)
                for i, entry in enumerate(algo_results[:loaded_count]):
                    setting_results[offset + i] = (
                        entry["learning_curve"],
                        entry["learning_curve_std"],
                        entry["timesteps"],
                    )
                    jobs[i]["curve_label"] = entry["curve_label"]
                if len(algo_results) > n_jobs:
                    print(
                        f"[{algo_upper}] Ignoring {len(algo_results) - n_jobs} extra matching sheet(s) "
                        f"because only {n_jobs} job(s) are configured."
                    )
                if len(algo_results) < n_jobs:
                    print(f"[{algo_upper}] Only {len(algo_results)}/{n_jobs} sheets matched. "
                          f"Running remaining {n_jobs - len(algo_results)} from scratch.")
                    for i in range(len(algo_results), n_jobs):
                        pending_settings.append((offset + i, jobs[i]))
                    algos_needing_save.add(algo_upper)
            else:
                mismatch_str = ""
                if _mismatches:
                    parts = []
                    for param, (sheet_val, cfg_val) in _mismatches.items():
                        cfg_display = (
                            str(list(cfg_val)) if isinstance(cfg_val, (list, np.ndarray))
                            else str(cfg_val)
                        )
                        parts.append(f"{param} (Disk data: {sheet_val}, Config: {cfg_display})")
                    mismatch_str = "  Mismatch reason(s): " + "; ".join(parts)
                print(f"[{algo_upper}] No matching sheets found in Excel. Re-running from scratch.{mismatch_str}")
                for i, job in enumerate(jobs):
                    pending_settings.append((offset + i, job))
                algos_needing_save.add(algo_upper)
        else:
            print(f"[{algo_upper}] Running experiments from scratch.\n")
            for i, job in enumerate(jobs):
                pending_settings.append((offset + i, job))
            algos_needing_save.add(algo_upper)

        offset += n_jobs

    # ── Pass 2: run all pending settings (not found in disk data) in one parallel pool ──
    if pending_settings:
        cpu_count = os.cpu_count() or 1
        total_tasks = len(pending_settings) * n_repetitions
        max_workers = min(total_tasks, cpu_count)
        print(f"CPU cores available: {cpu_count}. "
              f"Total tasks: {total_tasks} "
              f"({len(pending_settings)} setting(s) × {n_repetitions} rep(s)). "
              f"Parallel workers: {max_workers}.\n")
        for global_idx, job in pending_settings:
            print(f"Setting {global_idx + 1}/{len(all_setting_jobs)}: {job['curve_label']}")
        print()
        _run_pending_parallel(
            pending_settings=pending_settings,
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            max_train_episode_length=max_train_episode_length,
            max_eval_episode_length=max_eval_episode_length,
            base_seed=base_seed,
            setting_results=setting_results,
        )

    # ── Pass 3: save newly computed results per algo to excel files, algo_upper: algorithm name uppercased ──
    # Append any newly completed settings to the existing workbook using the
    # shared workbook formatting rules. Existing sheets are preserved and
    # matching signatures are skipped.
    for algo_upper in algos_needing_save:
        jobs = algo_jobs[algo_upper]
        off = algo_job_offsets[algo_upper]
        cfg = algo_configs_map[algo_upper]
        base_filename = algo_filenames[algo_upper]
        algo_results_to_save = [setting_results[off + i] for i in range(len(jobs))]
        if any(r is not None for r in algo_results_to_save):
            save_algorithm_workbook(
                data_sheets_dir,
                base_filename,
                algo_upper,
                jobs,
                algo_results_to_save,
                format_sheets=format_sheets,
            )

    # Collect all generated filenames for this run (used to skip in extra-curve loading)
    current_basenames = set(f"{fn}.xlsx" for fn in algo_filenames.values())

    # ── Also load all OTHER Excel files from the "data sheets" directory ──
    # Only files whose algo prefix (first word of filename) matches a selected
    # algorithm are loaded - this enforces include_<algo>_in_training = False.
    extra_curves = []
    if use_existing_disk_data:
        algo_configs = {k: v for k, v in algo_configs_map.items() if v is not None}
        all_disk_curves = _load_all_excel_curves(
            data_sheets_dir,
            algo_configs,
            formatted_sheets=formatted_sheets,
        )
        for curve_info in all_disk_curves:
            if curve_info["source_file"] in current_basenames:
                continue  # skip - already handled above
            source_algo = os.path.splitext(curve_info["source_file"])[0].upper()
            if source_algo not in algo_jobs:
                continue  # include_<algo>_in_training is False - skip
            extra_curves.append(curve_info)
        if extra_curves:
            from collections import Counter
            counts = Counter(c["source_file"] for c in extra_curves)
            for fname, n in sorted(counts.items()):
                print(f"Loaded {n} additional curve(s) from '{fname}' Excel file in '{data_sheets_dir}'.")

    # Update legend headroom for extra curves
    if extra_curves:
        total_entries = len(all_setting_jobs) + len(extra_curves) + 2
        legend_headroom = max(90, 80 + 20 * total_entries)
        for pc in plot_configs:
            pc["plot"].set_ylim(0, y_axis_episode_length_cap + legend_headroom)

    # ── 1. Plot current experiment settings across all smoothing windows ──
    for idx, job in enumerate(all_setting_jobs):
        lc_raw, lc_std_raw, timesteps = setting_results[idx]
        curve_label = job["curve_label"]
        for pc in plot_configs:
            window = int(pc["window"])
            lc_w = _apply_optional_smoothing(np.asarray(lc_raw, dtype=np.float32), window)
            lc_std_w = _apply_optional_smoothing(np.asarray(lc_std_raw, dtype=np.float32), window)
            y_cap = float(optimal_episode_return)
            lc_w = np.minimum(lc_w, y_cap)
            plot_obj = pc["plot"]
            plot_obj.add_curve(timesteps, lc_w, label=curve_label)
            if shade_ci:
                plot_obj.add_shaded_ci(
                    timesteps, lc_w, lc_std_w, n=n_repetitions,
                    alpha=curve_ci_alpha, fill_opacity=curve_shaded_area_opacity,
                    y_upper_cap=y_cap,
                )

    # ── 2. Plot extra curves from other Excel files ──
    for curve_info in extra_curves:
        lc_raw = curve_info["learning_curve"]
        lc_std_raw = curve_info["learning_curve_std"]
        timesteps = curve_info["timesteps"]
        curve_label = curve_info["curve_label"]
        for pc in plot_configs:
            window = int(pc["window"])
            lc_w = _apply_optional_smoothing(np.asarray(lc_raw, dtype=np.float32), window)
            lc_std_w = _apply_optional_smoothing(np.asarray(lc_std_raw, dtype=np.float32), window)
            y_cap = float(optimal_episode_return)
            lc_w = np.minimum(lc_w, y_cap)
            plot_obj = pc["plot"]
            plot_obj.add_curve(timesteps, lc_w, label=curve_label, ls="solid")
            if shade_ci and "n_repetitions" in curve_info:
                plot_obj.add_shaded_ci(
                    timesteps, lc_w, lc_std_w,
                    n=curve_info.get("n_repetitions", n_repetitions),
                    alpha=curve_ci_alpha, fill_opacity=curve_shaded_area_opacity,
                    y_upper_cap=y_cap,
                )

    # ── Add benchmark + optimum to each plot ──
    for pc in plot_configs:
        plot_obj = pc["plot"]
        plot_obj.ax.plot(
            pc["benchmark_steps"], pc["benchmark_returns"],
            label=benchmark_name, ls=":", c="gray",
        )
        plot_obj.add_hline(optimal_episode_return, label="CartPole optimum")

    plot_filename_tag = "-".join(e.upper() for e in experiments)
    
    #### Save all plots with a filename that includes the experiment names and smoothing window info
    for pc in plot_configs:
        window = int(pc["window"])
        suffix = f"w{window}-not-smoothed" if window <= 1 else f"w{window}-smoothed"
        pc["plot"].save(f"{plot_filename_tag}_{suffix}.png")

    # ── Optional animation (uses last REINFORCE episode if available) ──
    if animation_plot and "REINFORCE" in [e.upper() for e in experiments]:
        from REINFORCE import REINFORCE_Agent, run_reinforce
        from Agent import trained_nn_policy

        rc = reinforce_config or {}
        actor_hidden_nn = np.asarray(rc.get("actor_hidden_nn", np.array([32, 32])), dtype=np.int32)
        actor_lr = np.atleast_1d(np.asarray(rc.get("actor_lr", np.array([0.005])), dtype=np.float32))
        gamma = float(rc.get("gamma", 0.99))

        torch.manual_seed(base_seed)
        agent = REINFORCE_Agent(
            actor_hidden_nn=actor_hidden_nn,
            actor_lr=float(actor_lr[0]),
            gamma=gamma,
        )
        test_env = environ.CartPoleEnvironment(
            max_episode_length=max_train_episode_length, render_mode="rgb_array",
        )
        run_reinforce(
            agent,
            test_env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_train_episode_length,
        )
        anim = CartPoleAgentPlot(
            test_env, title="CartPole Agent Plot",
            curve_plot=curve_plot, animation_plot=animation_plot,
        )
        anim.test_one_episode(
            test_env,
            policy=lambda obs: trained_nn_policy(agent.actor, obs),
        )

    if curve_plot or animation_plot:
        plt.show()

    total_time = (time.perf_counter() - start_time) / 60.0
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Total execution time: {total_time:.3f} minutes\n")
    print(f"\nExperiment finished in {total_time:.3f} minutes.")




################[ Main Execution Block             ]################
if __name__ == "__main__":
    base_seed = 47
    env=environ.CartPoleEnvironment(max_episode_length=500, render_mode="rgb_array",seed=base_seed)
    plot=fn.CartPoleAgentPlot(env, title="Test CartPole Agent Plot", curve_plot=False)
    preview_animation = plot.test_one_episode(env=env, policy="test_policy")
    plt.show()
####################################################################


def _apply_optional_smoothing(learning_curve, plot_smoothing_window):
    if plot_smoothing_window is None:
        return learning_curve
    max_window = len(learning_curve) if len(learning_curve) % 2 == 1 else len(learning_curve) - 1
    window = min(int(plot_smoothing_window), max_window)
    if window >= 3:
        return smooth(learning_curve, window)
    return learning_curve


def _load_benchmark_curve(
    benchmark_curve,
    project_eval_interval,
    project_n_timesteps,
    benchmark_eval_interval=250,
    episode_return_column="Episode_Return",
):
    benchmark_files = {
        1: os.path.join("Baseline data", "BaselineDataCartPole_run1.csv"),
        2: os.path.join("Baseline data", "BaselineDataCartPole_run2.csv"),
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
    if not np.any(in_horizon):
        print(
            f"[benchmark] No benchmark points fall within project_n_timesteps={project_n_timesteps}; "
            "using the full benchmark curve instead."
        )
        return env_steps.astype(np.int32), returns

    env_steps = env_steps[in_horizon]
    returns = returns[in_horizon]
    return env_steps.astype(np.int32), returns


def average_over_repetitions(
    method,
    n_repetitions,
    n_timesteps,
    eval_interval,
    max_episode_length,
    actor_lr,
    gamma,
    actor_hidden_nn=np.array([16, 16]),
    critic_hidden_nn=np.array([64, 64]),
    critic_lr=0.001,
    base_seed=42,
    plot_smoothing_window=None,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
    return_raw=False,
):
    """Run ``n_repetitions`` of the given method and return (mean, std, timesteps)."""

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import Manager
    from Helper import _create_step_progress_bar
    import time as _time

    returns_over_repetitions = []
    timesteps = None

    parallel_workers = max(1, min(n_repetitions, os.cpu_count() or 1))
    use_parallel = parallel_workers > 1 and n_repetitions > 1

    if use_parallel:
        manager = Manager()
        step_counters = [manager.Value("i", 0) for _ in range(n_repetitions)]
        try:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                future_to_rep = {}
                for rep in range(n_repetitions):
                    run_seed = base_seed + rep
                    future = executor.submit(
                        _run_single_repetition,
                        method=method,
                        actor_hidden_nn=actor_hidden_nn,
                        critic_hidden_nn=critic_hidden_nn,
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        gamma=gamma,
                        max_episode_length=max_episode_length,
                        n_timesteps=n_timesteps,
                        eval_interval=eval_interval,
                        run_seed=run_seed,
                        rep_index=rep,
                        n_repetitions=n_repetitions,
                        enable_progress_bar=False,
                        shared_step_counter=step_counters[rep],
                        eval_with_env_episode_trials=eval_with_env_episode_trials,
                        n_eval_episodes=n_eval_episodes,
                    )
                    future_to_rep[future] = rep

                pbars = [
                    _create_step_progress_bar(
                        total=n_timesteps,
                        desc=f"{method.upper()} Rep {rep + 1}/{n_repetitions}",
                        position=rep,
                        leave=True,
                    )
                    for rep in range(n_repetitions)
                ]

                done_futures = set()
                try:
                    while len(done_futures) < n_repetitions:
                        for rep in range(n_repetitions):
                            current = step_counters[rep].value
                            delta = current - pbars[rep].n
                            if delta > 0:
                                pbars[rep].update(delta)

                        for future in list(future_to_rep):
                            if future not in done_futures and future.done():
                                done_futures.add(future)
                                rep = future_to_rep[future]
                                rep_returns, rep_timesteps = future.result()
                                returns_over_repetitions.append(np.asarray(rep_returns, dtype=np.float32))
                                if timesteps is None:
                                    timesteps = np.asarray(rep_timesteps, dtype=np.int32)
                                remaining = n_timesteps - pbars[rep].n
                                if remaining > 0:
                                    pbars[rep].update(remaining)

                        _time.sleep(0.25)
                finally:
                    for pb in pbars:
                        pb.close()
                    print()
        finally:
            manager.shutdown()
    else:
        for rep in range(n_repetitions):
            run_seed = base_seed + rep
            rep_returns, rep_timesteps = _run_single_repetition(
                method=method,
                actor_hidden_nn=actor_hidden_nn,
                critic_hidden_nn=critic_hidden_nn,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                max_episode_length=max_episode_length,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                run_seed=run_seed,
                rep_index=rep,
                n_repetitions=n_repetitions,
                enable_progress_bar=True,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
            )
            returns_over_repetitions.append(np.asarray(rep_returns, dtype=np.float32))
            if timesteps is None:
                timesteps = np.asarray(rep_timesteps, dtype=np.int32)

    min_length = min(len(rep_returns) for rep_returns in returns_over_repetitions)
    returns_over_repetitions = [
        np.asarray(rep_returns[:min_length], dtype=np.float32)
        for rep_returns in returns_over_repetitions
    ]
    if timesteps is not None:
        timesteps = np.asarray(timesteps[:min_length], dtype=np.int32)

    all_returns = np.array(returns_over_repetitions)
    learning_curve = np.mean(all_returns, axis=0)
    learning_curve_std = (
        np.std(all_returns, axis=0, ddof=1)
        if all_returns.shape[0] > 1
        else np.zeros_like(learning_curve)
    )
    learning_curve = _apply_optional_smoothing(learning_curve, plot_smoothing_window)
    learning_curve_std = _apply_optional_smoothing(learning_curve_std, plot_smoothing_window)

    if return_raw:
        raw_returns = np.asarray(returns_over_repetitions, dtype=np.float32)
        return learning_curve, learning_curve_std, timesteps, raw_returns
    return learning_curve, learning_curve_std, timesteps


def _run_single_repetition(
    method,
    actor_hidden_nn,
    critic_hidden_nn=np.array([64, 64]),
    actor_lr=0.001,
    critic_lr=0.001,
    gamma=0.99,
    max_episode_length=500,
    max_train_episode_length=None,
    max_eval_episode_length=None,
    n_timesteps=1000000,
    eval_interval=250,
    run_seed=42,
    rep_index=0,
    n_repetitions=1,
    enable_progress_bar=True,
    shared_step_counter=None,
    n_agent_state_elements=4,
    n_actions=2,
    TN_step=10,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
):
    """Run one training repetition (pickle-safe for ProcessPoolExecutor)."""

    import torch
    import Environment as environ

    torch.manual_seed(run_seed)

    # Backward/forward compatible mapping:
    # - max_episode_length historically meant "training truncation"
    # - new args separate training vs evaluation truncation lengths
    if max_train_episode_length is None:
        max_train_episode_length = max_episode_length
    if max_eval_episode_length is None:
        max_eval_episode_length = max_train_episode_length

    if method == "REINFORCE":
        from REINFORCE import REINFORCE_Agent, run_reinforce

        agent = REINFORCE_Agent(
            actor_hidden_nn=actor_hidden_nn,
            actor_lr=actor_lr,
            gamma=gamma,
            use_critic=False,
        )
        env = environ.CartPoleEnvironment(
            max_episode_length=max_train_episode_length,
            render_mode="rgb_array",
        )
        rep_returns, rep_timesteps = run_reinforce(
            agent,
            env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_train_episode_length,
            max_eval_episode_length=max_eval_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"REINFORCE Rep {rep_index + 1}/{n_repetitions}",
            shared_step_counter=shared_step_counter,
            eval_with_env_episode_trials=eval_with_env_episode_trials,
            n_eval_episodes=n_eval_episodes,
        )
    elif method == "ac":
        from AC import AC_Agent, run_ac

        agent = AC_Agent(
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
        )
        env = environ.CartPoleEnvironment(
            max_episode_length=max_train_episode_length,
            render_mode="rgb_array",
        )
        rep_returns, rep_timesteps = run_ac(
            agent,
            env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            max_eval_episode_length=max_eval_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"AC Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            eval_with_env_episode_trials=eval_with_env_episode_trials,
            n_eval_episodes=n_eval_episodes,
        )
    elif method == "a2c":
        from A2C import A2C_Agent, run_a2c

        agent = A2C_Agent(
            critic_lr=critic_lr,
            critic_hidden_nn=critic_hidden_nn,
            TN_step=TN_step,
            n_actions=n_actions,
            n_agent_state_elements=n_agent_state_elements,
            actor_hidden_nn=actor_hidden_nn,
            actor_lr=actor_lr,
            gamma=gamma,
        )
        env = environ.CartPoleEnvironment(
            max_episode_length=max_train_episode_length,
            render_mode="rgb_array",
        )
        rep_returns, rep_timesteps = run_a2c(
            agent,
            env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_train_episode_length,
            max_eval_episode_length=max_eval_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"A2C Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            eval_with_env_episode_trials=eval_with_env_episode_trials,
            n_eval_episodes=n_eval_episodes,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return rep_returns, rep_timesteps
