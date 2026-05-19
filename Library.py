import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as visplt
import matplotlib.animation
import torch
import torch.nn as nn
import numpy as np
import Environment as environ
import Library as fn
import os
import time
from datetime import datetime
from Helper import smooth, RUN_TIMESTAMP
from scipy.stats import t as t_dist

# Begin Class LearningCurvePlot ##############################################################
class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, x, y, label=None, ls="solid", color=None):
        """y: vector of average reward results
        label: string to appear as label in plot legend
        """
        plot_kwargs = {"ls": ls}
        if color is not None:
            plot_kwargs["color"] = color
        if label is not None:
            self.ax.plot(x, y, label=label, **plot_kwargs)
        else:
            self.ax.plot(x, y, **plot_kwargs)

    def add_shaded_ci(
        self,
        x,
        y_mean,
        y_std,
        n,
        alpha=0.2,
        fill_opacity=0.15,
        y_lower_cap=None,
        y_upper_cap=None,
        color=None,
    ):
        """Add a shaded confidence band around the mean curve.
        alpha controls CI significance (e.g., 0.05 for 95% CI),
        fill_opacity controls the visual transparency of the shaded area.
        """
        t_crit = t_dist.ppf(1 - alpha / 2, df=max(n - 1, 1))
        margin = t_crit * y_std / np.sqrt(max(n, 1))
        y_lower = y_mean - margin
        y_upper = y_mean + margin
        if y_lower_cap is not None:
            y_lower = np.maximum(y_lower, y_lower_cap)
        if y_upper_cap is not None:
            y_upper = np.minimum(y_upper, y_upper_cap)
        if color is None:
            color = self.ax.get_lines()[-1].get_color()  # match the last plotted line
        self.ax.fill_between(x, y_lower, y_upper, alpha=fill_opacity, color=color)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    @staticmethod
    def _resolve_non_overwriting_path(output_path: str) -> str:
        """Return output_path if it doesn't exist; otherwise add (1), (2), ... before the extension."""
        if not os.path.exists(output_path):
            return output_path

        root, ext = os.path.splitext(output_path)
        i = 1
        while True:
            candidate = f"{root} ({i}){ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def save(self, name='test.png') -> str:
        """name: string for filename of saved figure.

        If the target filename already exists, saves to an enumerated name:
        file.png -> file (1).png -> file (2).png -> ...
        """
        self.ax.legend(
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.3,
        )
        self.fig.tight_layout()

        output_path = name
        if not os.path.isabs(name):
            os.makedirs("plots", exist_ok=True)
            output_path = os.path.join("plots", os.path.basename(name))
        else:
            # Ensure parent folder exists for absolute paths
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        root, ext = os.path.splitext(output_path)
        output_path = f"{root}_{RUN_TIMESTAMP}{ext}"

        output_path = self._resolve_non_overwriting_path(output_path)
        self.fig.savefig(output_path, dpi=300)
        return output_path
# End Class LearningCurvePlot ##############################################################

def argmax(x):
    """Own variant of np.argmax with random tie breaking."""
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except Exception:
        return np.argmax(x)

def egreedy(Qa_s, eps):
    """Sample one action using epsilon-greedy policy.
    Qa_s: 1D array of Q-values for current state's actions
    eps: epsilon in the closed boundary [0,1]
    """
    n_A = len(Qa_s)     # number of actions
    greedy_a = argmax(Qa_s)  # tie breaking argmax()
    # Base probability for all actions, fill probs matrix with the same values (will not sum up to 1 yet)
    probs = np.full(n_A, eps / n_A, dtype=float)
    # Greedy action gets the remaining probability mass (1 - eps) plus its share of the exploration probability (eps/n_A)
    probs[greedy_a] = 1.0 - eps * (n_A - 1) / n_A
    selected_action = np.random.choice(n_A, p=probs)
    # Sample action from this distribution
    return selected_action

def softmax(x, temp):
    """Computes the softmax of vector x with temperature parameter 'temp'."""
    x = x / temp  # scale by temperature
    z = x - max(x)  # subtract max to prevent overflow of softmax
    probs = np.exp(z) / np.sum(np.exp(z))  # compute softmax
    selected_action = np.random.choice(len(x), p=probs)  # Sample action from
    return int(selected_action)



# Begin Class CartPoleAgentPlot ##############################################################
class CartPoleAgentPlot:
    ''' Class for plotting CartPole agent behavior during training '''
    def __init__(self, env, title=None, show_curve_plots=False, animation_plot=False):
        self.env = env
        if show_curve_plots:
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
    REINFORCE_config=None,
    AC_config=None,
    A2C_config=None,
    DQN_config=None,
    PPO_config=None,
):
    """Orchestrate training, data loading, and plotting for all selected experiments.

    Parameters
    ----------
    experiments : list[str]
        Algorithm names to run. Supported: "REINFORCE", "AC", "A2C", "DQN", "PPO".
    global_config : dict
        Global/shared parameters (benchmark, plotting, environment, seed).
    REINFORCE_config : dict or None
        REINFORCE-specific hyperparameters. Required when "REINFORCE" in experiments.
    AC_config : dict or None
        AC-specific hyperparameters. Required when "AC" in experiments.
    A2C_config : dict or None
        A2C-specific hyperparameters. Required when "A2C" in experiments.
    DQN_config : dict or None
        DQN-specific hyperparameters. Required when "DQN" in experiments.
    PPO_config : dict or None
        PPO-specific hyperparameters. Required when "PPO" in experiments.
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
    use_existing_disk_trained_networks = bool(gc.get("use_existing_disk_trained_networks", False))
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
    unused_cpu_cores = int(gc.get("UNUSED_CPU_CORES", 0))
    if unused_cpu_cores < 0:
        unused_cpu_cores = 0
    # Keep the *original* show_curve_plots value for the matplotlib "block=" behavior;
    # show_individual_plots is the derived flag that can be turned off when the
    # combined preview replaces the individual curve plots.
    show_curve_plots = bool(gc.get("show_curve_plots", False))
    show_individual_plots = show_curve_plots
    animation_plot = bool(gc.get("animation_plot", False))
    separate_algorithm_plots = bool(gc.get("separate_algorithm_plots", False))

    start_time = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Experiment started at: {start_human}\n")
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Start the process at: {start_human}\n")
    print(f"Included experiments: {', '.join(experiments)}\n")

    from Helper import (
        _build_a2c_jobs,
        _build_ac_jobs,
        _build_algo_filename,
        _build_dqn_jobs,
        _build_ppo_jobs,
        _build_reinforce_jobs,
        _load_all_excel_curves,
        _match_sheets_to_jobs,
        _run_pending_parallel,
        build_returns_summary_table,
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

    # User request: only show the combined 2-subplot plot during execution.
    # If the required windows exist (101 and 201), disable other plot display (curve figures + animation).
    # In separate-plots mode each algo gets its own per-window figures shown live, so we suppress the combined preview.
    desired_windows = [101, 201]
    available_windows = {int(w) for w in plot_smoothing_windows}
    will_have_combined_preview = (
        (not separate_algorithm_plots)
        and all(w in available_windows for w in desired_windows)
    )
    if will_have_combined_preview:
        show_individual_plots = False
        animation_plot = False

    # ── Build setting jobs for each selected algorithm (grouped by algo) ──
    algo_jobs = {}  # algo_upper -> list of setting_jobs
    algo_configs_map = {
        "REINFORCE": REINFORCE_config,
        "AC": AC_config,
        "A2C": A2C_config,
        "DQN": DQN_config,
        "PPO": PPO_config,
    }
    # Preserve the original (un-augmented) algo configs for workbook-level
    # meta validation. The augmented copies merge in global keys for per-sheet
    # hyperparameter matching, but the meta sheet stores the user-supplied
    # algo config as it appears in Experiment.py.
    original_algo_configs_map = {
        algo_key: cfg for algo_key, cfg in algo_configs_map.items() if cfg is not None
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
            if REINFORCE_config is None:
                raise ValueError("REINFORCE_config dict is required when REINFORCE is included.")
            jobs = _build_reinforce_jobs(
                algo_config=REINFORCE_config,
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
            if AC_config is None:
                raise ValueError("AC_config dict is required when AC is included.")
            jobs = _build_ac_jobs(
                algo_config=AC_config,
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
            if A2C_config is None:
                raise ValueError("A2C_config dict is required when A2C is included.")
            jobs = _build_a2c_jobs(
                algo_config=A2C_config,
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
            if DQN_config is None:
                raise ValueError("DQN_config dict is required when DQN is included.")
            jobs = _build_dqn_jobs(
                dqn_config=DQN_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
            )
        elif algo_upper == "PPO":
            if PPO_config is None:
                raise ValueError("PPO_config dict is required when PPO is included.")
            jobs = _build_ppo_jobs(
                algo_config=PPO_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_train_episode_length=max_train_episode_length,
                max_eval_episode_length=max_eval_episode_length,
                base_seed=base_seed,
                eval_with_env_episode_trials=eval_with_env_episode_trials,
                n_eval_episodes=n_eval_episodes,
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

    # ── Prepare plot config(s) ──
    # In combined mode (default): one shared figure per smoothing window across all algorithms.
    # In separate_algorithm_plots mode: one figure per (algorithm, smoothing window) pair so that
    # each algo's plots can be finalized and shown the moment that algo finishes executing.
    title_tag = " + ".join(experiments)

    def _build_pc_for_title(title_prefix: str, create_figures: bool = True) -> list[dict]:
        cfgs = []
        for window in plot_smoothing_windows:
            window = int(window)
            is_not_smoothed = window <= 1
            benchmark_returns = _apply_optional_smoothing(
                np.asarray(benchmark_returns_raw, dtype=np.float32), window
            )
            benchmark_returns = np.minimum(benchmark_returns, float(optimal_episode_return))
            suffix_label = "not smoothed plot" if is_not_smoothed else "smoothed plot"
            title_str = f"{title_prefix} - {suffix_label}"
            # In separate mode, defer figure creation to _finalize_algo_plot so that
            # not-yet-populated algo windows don't pop up as empty placeholders the
            # moment another algo's plot is shown non-blocking.
            plot_obj = LearningCurvePlot(title=title_str) if create_figures else None
            cfgs.append({
                "window": window,
                "is_not_smoothed": is_not_smoothed,
                "title": title_str,
                "plot": plot_obj,
                "benchmark_steps": benchmark_steps,
                "benchmark_returns": benchmark_returns,
            })
        return cfgs

    if separate_algorithm_plots:
        algo_plot_configs: dict[str, list[dict]] = {
            algo_upper: _build_pc_for_title(algo_upper, create_figures=False)
            for algo_upper in algo_jobs.keys()
        }
        # plot_configs kept for shared code paths that iterate "all current plots"
        plot_configs = [pc for cfgs in algo_plot_configs.values() for pc in cfgs]
    else:
        algo_plot_configs = {}
        plot_configs = _build_pc_for_title(title_tag)

    # ── Allocate top headroom for legend (per-algo headroom is set inside
    #    _finalize_algo_plot in separate mode where the figures don't exist yet). ──
    if not separate_algorithm_plots:
        expected_legend_entries = len(all_setting_jobs) + 2
        legend_headroom = max(90, 80 + 20 * expected_legend_entries) * y_axis_episode_length_cap / 400.0
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
    algo_added_counts: dict[str, int] = {algo_upper: 0 for algo_upper in algo_jobs.keys()}

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
                aligned_results, _mismatches = _match_sheets_to_jobs(
                    excel_path,
                    jobs,
                    formatted_sheets=formatted_sheets,
                    global_config=gc,
                    algo_config=cfg,
                )
            except Exception as exc:
                print(f"[{algo_upper}] Existing Excel data is incompatible. Re-running from scratch. Reason: {exc}")
                aligned_results = [None] * n_jobs
                _mismatches = {}

            matched_count = sum(1 for r in aligned_results if r is not None)

            if matched_count > 0:
                print(f"[{algo_upper}] Loaded {matched_count}/{n_jobs} matching setting(s) from: {excel_path}")

            for i, entry in enumerate(aligned_results):
                if entry is None:
                    pending_settings.append((offset + i, jobs[i]))
                else:
                    setting_results[offset + i] = (
                        entry["learning_curve"],
                        entry["learning_curve_std"],
                        entry["timesteps"],
                        entry.get("raw_returns"),
                    )
                    jobs[i]["curve_label"] = entry["curve_label"]

            if matched_count < n_jobs:
                algos_needing_save.add(algo_upper)
                if matched_count == 0:
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
                    print(f"[{algo_upper}] No matching sheets found in Excel. Re-running all {n_jobs} from scratch.{mismatch_str}")
                else:
                    print(
                        f"[{algo_upper}] {n_jobs - matched_count}/{n_jobs} setting(s) not on disk. "
                        f"Running those from scratch."
                    )
        else:
            print(f"[{algo_upper}] Running experiments from scratch.\n")
            for i, job in enumerate(jobs):
                pending_settings.append((offset + i, job))
            algos_needing_save.add(algo_upper)

        offset += n_jobs

    # ── Separate-plots bookkeeping: precompute extras (so per-algo plots can include them)
    #    and define the per-algo finalize helper that saves + (optionally) shows live.
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    extra_curves_early: list[dict] = []
    if separate_algorithm_plots and use_existing_disk_data:
        algo_configs_for_extras = {k: v for k, v in algo_configs_map.items() if v is not None}
        all_disk_curves_early = _load_all_excel_curves(
            data_sheets_dir,
            algo_configs_for_extras,
            formatted_sheets=formatted_sheets,
            global_config=gc,
            original_algo_configs=original_algo_configs_map,
        )
        current_basenames_pre = set(f"{fn}.xlsx" for fn in algo_filenames.values())
        for curve_info in all_disk_curves_early:
            if curve_info["source_file"] in current_basenames_pre:
                continue
            source_algo = os.path.splitext(curve_info["source_file"])[0].upper()
            if source_algo not in algo_jobs:
                continue
            extra_curves_early.append(curve_info)
        if extra_curves_early:
            from collections import Counter
            counts = Counter(c["source_file"] for c in extra_curves_early)
            for fname, n in sorted(counts.items()):
                print(f"Loaded {n} additional curve(s) from '{fname}' Excel file in '{data_sheets_dir}'.")

    algo_pending_count: dict[str, int] = {algo_upper: 0 for algo_upper in algo_jobs.keys()}
    for global_idx, _job in pending_settings:
        for _algo_upper, _off in algo_job_offsets.items():
            _n = len(algo_jobs[_algo_upper])
            if _off <= global_idx < _off + _n:
                algo_pending_count[_algo_upper] += 1
                break
    algo_finalized: set[str] = set()
    saved_paths_by_window: dict[int, str] = {}

    def _finalize_algo_plot(algo_upper: str) -> None:
        """Populate, save, and (optionally) display one algo's per-window plots."""
        if algo_upper in algo_finalized:
            return
        cfgs = algo_plot_configs[algo_upper]
        off_a = algo_job_offsets[algo_upper]
        jobs_a = algo_jobs[algo_upper]

        # Lazily create this algo's figures now (separate mode defers creation so
        # other not-yet-finalized algos' empty figures don't pop up when we call
        # plt.show(block=False) / plt.pause for THIS algo).
        for pc in cfgs:
            if pc.get("plot") is None:
                pc["plot"] = LearningCurvePlot(title=pc["title"])

        extras_for_algo = [
            c for c in extra_curves_early
            if os.path.splitext(c["source_file"])[0].upper() == algo_upper
        ]
        entries = len(jobs_a) + len(extras_for_algo) + 2
        headroom = max(90, 80 + 20 * entries) * y_axis_episode_length_cap / 400.0
        for pc in cfgs:
            pc["plot"].set_ylim(0, y_axis_episode_length_cap + headroom)

        for i, job in enumerate(jobs_a):
            entry = setting_results[off_a + i]
            if entry is None:
                continue
            lc_raw, lc_std_raw, timesteps = entry[:3]
            curve_label = job["curve_label"]
            for pc in cfgs:
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

        for curve_info in extras_for_algo:
            lc_raw = curve_info["learning_curve"]
            lc_std_raw = curve_info["learning_curve_std"]
            timesteps = curve_info["timesteps"]
            curve_label = curve_info["curve_label"]
            for pc in cfgs:
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

        for pc in cfgs:
            plot_obj = pc["plot"]
            plot_obj.ax.plot(
                pc["benchmark_steps"], pc["benchmark_returns"],
                label=benchmark_name, ls=":", c="gray",
            )
            plot_obj.add_hline(optimal_episode_return, label="CartPole optimum")

        saved = 0
        for pc in cfgs:
            window = int(pc["window"])
            suffix = f"w{window}-not-smoothed" if window <= 1 else f"w{window}-smoothed"
            filename = f"{algo_upper}_{suffix}.png"
            output_path = os.path.join(plots_dir, filename)
            saved_path = pc["plot"].save(output_path)
            saved_paths_by_window[window] = saved_path
            saved += 1

        if show_curve_plots:
            try:
                plt.show(block=False)
                plt.pause(0.001)
            except Exception as exc:
                print(f"[plot] Could not display {algo_upper} plot(s) non-blocking: {exc}")
        print(f"[{algo_upper}] Saved {saved} plot(s) to {plots_dir}/")
        algo_finalized.add(algo_upper)

    if separate_algorithm_plots:
        # Algos with no pending settings are already fully loaded from disk -> finalize now.
        for _algo_upper in list(algo_jobs.keys()):
            if algo_pending_count[_algo_upper] == 0:
                _finalize_algo_plot(_algo_upper)

    def _on_setting_complete(global_idx: int) -> None:
        if not separate_algorithm_plots:
            return
        for _algo_upper, _off in algo_job_offsets.items():
            _n = len(algo_jobs[_algo_upper])
            if _off <= global_idx < _off + _n:
                algo_pending_count[_algo_upper] -= 1
                if algo_pending_count[_algo_upper] == 0:
                    _finalize_algo_plot(_algo_upper)
                return

    def _poll_callback() -> None:
        if separate_algorithm_plots and show_curve_plots:
            try:
                plt.pause(0.001)
            except Exception:
                pass

    # ── Pass 2: run all pending settings (not found in disk data) in one parallel pool ──
    if pending_settings:
        cpu_count = os.cpu_count() or 1
        total_tasks = len(pending_settings) * n_repetitions
        available_cpus = max(1, cpu_count - unused_cpu_cores)
        max_workers = min(total_tasks, available_cpus)
        print(
            f"CPU cores available: {cpu_count} "
            f"(reserving UNUSED_CPU_CORES={unused_cpu_cores} => using {available_cpus}). "
            f"Total tasks: {total_tasks} "
            f"({len(pending_settings)} setting(s) × {n_repetitions} rep(s)). "
            f"Parallel workers: {max_workers}.\n"
        )
        for global_idx, job in pending_settings:
            print(f"Setting {global_idx + 1}/{len(all_setting_jobs)}: {job['curve_label']}")
        print()

        # Report which network checkpoint files will be loaded by the workers.
        if use_existing_disk_trained_networks:
            from Checkpointing import (
                pg_actor_checkpoint_path,
                pg_critic_checkpoint_path,
            )
            pg_method_to_algo = {
                "REINFORCE": ("REINFORCE", False),
                "ac": ("AC", True),
                "a2c": ("A2C", True),
                "ppo": ("PPO", True),
            }
            reported_paths: set[str] = set()
            any_reported = False
            for _global_idx, job in pending_settings:
                method = job["method"]
                if method not in pg_method_to_algo:
                    continue
                algo_type, has_critic = pg_method_to_algo[method]
                kw = job["kwargs"]
                actor_path = pg_actor_checkpoint_path(
                    algo_type=algo_type,
                    actor_hidden_nn=kw["actor_hidden_nn"],
                ).file_path
                if actor_path not in reported_paths:
                    reported_paths.add(actor_path)
                    if os.path.isfile(actor_path):
                        print(f"[{algo_type}] Loading existing actor checkpoint: {actor_path}")
                    else:
                        print(f"[{algo_type}] No existing actor checkpoint at: {actor_path} (training from scratch)")
                    any_reported = True
                if has_critic:
                    critic_path = pg_critic_checkpoint_path(
                        algo_type=algo_type,
                        critic_hidden_nn=kw["critic_hidden_nn"],
                    ).file_path
                    if critic_path not in reported_paths:
                        reported_paths.add(critic_path)
                        if os.path.isfile(critic_path):
                            print(f"[{algo_type}] Loading existing critic checkpoint: {critic_path}")
                        else:
                            print(f"[{algo_type}] No existing critic checkpoint at: {critic_path} (training from scratch)")
                        any_reported = True
            if any_reported:
                print()

        _run_pending_parallel(
            pending_settings=pending_settings,
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            max_train_episode_length=max_train_episode_length,
            max_eval_episode_length=max_eval_episode_length,
            base_seed=base_seed,
            use_existing_disk_trained_networks=use_existing_disk_trained_networks,
            setting_results=setting_results,
            unused_cpu_cores=unused_cpu_cores,
            on_setting_complete=_on_setting_complete,
            poll_callback=_poll_callback,
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
            _filepath, added_count = save_algorithm_workbook(
                data_sheets_dir,
                base_filename,
                algo_upper,
                jobs,
                algo_results_to_save,
                format_sheets=format_sheets,
                verbose=False,
                global_config=gc,
                algo_config=original_algo_configs_map.get(algo_upper),
            )
            algo_added_counts[algo_upper] = int(added_count)

    combined_fig_shown = False
    if separate_algorithm_plots:
        # Per-algo plots were populated, saved, and (optionally) shown live in
        # _finalize_algo_plot as each algorithm finished. Nothing more to do here.
        pass
    else:
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
                global_config=gc,
                original_algo_configs=original_algo_configs_map,
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
                pc["plot"].set_ylim(0, y_axis_episode_length_cap + legend_headroom) * 1.4

        # ── 1. Plot current experiment settings across all smoothing windows ──
        for idx, job in enumerate(all_setting_jobs):
            lc_raw, lc_std_raw, timesteps = setting_results[idx][:3]
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
        close_individual_plot_figs = (not show_individual_plots) and (not animation_plot)

        new_plot_count = 0
        for pc in plot_configs:
            window = int(pc["window"])
            suffix = f"w{window}-not-smoothed" if window <= 1 else f"w{window}-smoothed"
            filename = f"{plot_filename_tag}_{suffix}.png"
            output_path = os.path.join(plots_dir, filename)
            saved_path = pc["plot"].save(output_path)
            saved_paths_by_window[window] = saved_path
            new_plot_count += 1
            if close_individual_plot_figs and hasattr(pc["plot"], "fig"):
                plt.close(pc["plot"].fig)

        print(f"Saved {new_plot_count} new plot(s) to plots/")

        # ── Combined display: smoothing windows 101 and 201 (side-by-side) ──
        desired_windows = [101, 201]
        available_windows = {int(pc["window"]) for pc in plot_configs}

        if all(w in available_windows for w in desired_windows):
            try:
                combined_fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                combined_fig.suptitle(f"Twin_{plot_filename_tag} (w=101 and w=201)")

                for ax, w in zip(axes, desired_windows):
                    saved_path = saved_paths_by_window.get(w)
                    if saved_path is not None and os.path.isfile(saved_path):
                        img = plt.imread(saved_path)
                        ax.imshow(img)
                        ax.axis("off")
                    else:
                        ax.axis("off")
                    ax.set_title(f"window {w}")

                plt.tight_layout()
                combined_fig_shown = True
            except Exception as exc:
                print(f"[plot] Failed to create combined subplot preview: {exc}")
                combined_fig_shown = False

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
            show_curve_plots=show_individual_plots, animation_plot=animation_plot,
        )
        anim.test_one_episode(
            test_env,
            policy=lambda obs: trained_nn_policy(agent.actor, obs),
        )

    if show_individual_plots or animation_plot or combined_fig_shown:
        plt.show(block=show_curve_plots)

    # ── Final summary: mean & std of returns per (algorithm, setting) ──
    try:
        build_returns_summary_table(
            algo_jobs=algo_jobs,
            setting_results=setting_results,
            algo_job_offsets=algo_job_offsets,
            n_repetitions=n_repetitions,
            last_fraction=0.1,
            output_dir=".",
        )
    except Exception as exc:
        print(f"[summary] Failed to build returns summary table: {exc}")

    total_time = (time.perf_counter() - start_time) / 60.0
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Total execution time: {total_time:.3f} minutes\n")
    print(f"\nExperiment finished in {total_time:.3f} minutes.")




################[ Main Execution Block             ]################
if __name__ == "__main__":
    base_seed = 47
    env=environ.CartPoleEnvironment(max_episode_length=500, render_mode="rgb_array",seed=base_seed)
    plot=fn.CartPoleAgentPlot(env, title="Test CartPole Agent Plot", show_curve_plots=False)
    preview_animation = plot.test_one_episode(env=env, policy="test_policy")
    plt.show()
####################################################################


def _apply_optional_smoothing(learning_curve, plot_smoothing_window):
    from Helper import _apply_optional_smoothing as _impl
    return _impl(learning_curve, plot_smoothing_window)


def _load_benchmark_curve(
    benchmark_curve,
    project_eval_interval,
    project_n_timesteps,
    benchmark_eval_interval=250,
    episode_return_column="Episode_Return",
):
    from Helper import _load_benchmark_curve as _impl
    return _impl(
        benchmark_curve=benchmark_curve,
        project_eval_interval=project_eval_interval,
        project_n_timesteps=project_n_timesteps,
        benchmark_eval_interval=benchmark_eval_interval,
        episode_return_column=episode_return_column,
    )


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
    unused_cpu_cores: int = 0,
):
    """Run 'n_repetitions' of the given method and return (mean, std, timesteps)."""

    from Helper import average_over_repetitions as _impl
    return _impl(
        method=method,
        n_repetitions=n_repetitions,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        max_episode_length=max_episode_length,
        actor_lr=actor_lr,
        gamma=gamma,
        actor_hidden_nn=actor_hidden_nn,
        critic_hidden_nn=critic_hidden_nn,
        critic_lr=critic_lr,
        base_seed=base_seed,
        plot_smoothing_window=plot_smoothing_window,
        eval_with_env_episode_trials=eval_with_env_episode_trials,
        n_eval_episodes=n_eval_episodes,
        return_raw=return_raw,
        unused_cpu_cores=unused_cpu_cores,
    )


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
    use_existing_disk_trained_networks: bool = False,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
    # PPO-specific
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 10,
    rollout_steps: int = 2048,
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
        from Checkpointing import (
            load_state_dict_if_present,
            pg_actor_checkpoint_path,
            save_state_dict_overwrite,
        )

        agent = REINFORCE_Agent(
            actor_hidden_nn=actor_hidden_nn,
            actor_lr=actor_lr,
            gamma=gamma,
            use_critic=False,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="REINFORCE",
            actor_hidden_nn=actor_hidden_nn,
        )
        if use_existing_disk_trained_networks:
            load_state_dict_if_present(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
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

        if rep_index == 0:
            save_state_dict_overwrite(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
            )

    elif method == "ac":
        from AC import AC_Agent, run_ac
        from Checkpointing import (
            load_state_dict_if_present,
            pg_actor_checkpoint_path,
            pg_critic_checkpoint_path,
            save_state_dict_overwrite,
        )

        agent = AC_Agent(
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="AC",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="AC",
            critic_hidden_nn=critic_hidden_nn,
        )

        if use_existing_disk_trained_networks:
            load_state_dict_if_present(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
            )
            load_state_dict_if_present(
                model=agent.critic,
                checkpoint_path=critic_ck.file_path,
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

        if rep_index == 0:
            save_state_dict_overwrite(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
            )
            save_state_dict_overwrite(
                model=agent.critic,
                checkpoint_path=critic_ck.file_path,
            )

    elif method == "a2c":
        from A2C import A2C_Agent, run_a2c
        from Checkpointing import (
            load_state_dict_if_present,
            pg_actor_checkpoint_path,
            pg_critic_checkpoint_path,
            save_state_dict_overwrite,
        )

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

        actor_ck = pg_actor_checkpoint_path(
            algo_type="A2C",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="A2C",
            critic_hidden_nn=critic_hidden_nn,
        )

        if use_existing_disk_trained_networks:
            load_state_dict_if_present(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
            )
            load_state_dict_if_present(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
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
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"A2C Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            max_eval_episode_length=max_eval_episode_length,
            eval_with_env_episode_trials=eval_with_env_episode_trials,
            n_eval_episodes=n_eval_episodes,
        )

        if rep_index == 0:
            save_state_dict_overwrite(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
            )
            save_state_dict_overwrite(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
            )

    elif method == "ppo":
        from PPO import PPO_Agent, run_ppo
        from Checkpointing import (
            load_state_dict_if_present,
            pg_actor_checkpoint_path,
            pg_critic_checkpoint_path,
            save_state_dict_overwrite,
        )

        agent = PPO_Agent(
            n_agent_state_elements=n_agent_state_elements,
            n_actions=n_actions,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            n_epochs=n_epochs,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="PPO",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="PPO",
            critic_hidden_nn=critic_hidden_nn,
        )

        if use_existing_disk_trained_networks:
            load_state_dict_if_present(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
            )
            load_state_dict_if_present(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
            )

        env = environ.CartPoleEnvironment(
            max_episode_length=max_train_episode_length,
            render_mode="rgb_array",
        )
        rep_returns, rep_timesteps = run_ppo(
            agent,
            env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_train_episode_length,
            rollout_steps=rollout_steps,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"PPO Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            max_eval_episode_length=max_eval_episode_length,
            eval_with_env_episode_trials=eval_with_env_episode_trials,
            n_eval_episodes=n_eval_episodes,
        )

        if rep_index == 0:
            save_state_dict_overwrite(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
            )
            save_state_dict_overwrite(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
            )

    return rep_returns, rep_timesteps
