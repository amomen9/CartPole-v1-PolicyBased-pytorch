import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as visplt
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
import Environment as environ
import Library as fn
import os
import time
from datetime import datetime
from Helper import smooth, RUN_TIMESTAMP, _create_step_progress_bar
from scipy.stats import t as t_dist

# Begin Class LearningCurvePlot ##############################################################
class LearningCurvePlot:

    def __init__(self, title=None, subtitle=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')
        if title is not None:
            self.fig.suptitle(title, fontsize=14, y=0.98)
        if subtitle is not None:
            self.fig.text(
                0.5,
                0.925,
                subtitle,
                ha="center",
                va="top",
                fontsize=10,
            )

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


def _latest_checkpoint_path(base_path):
    dir_path = os.path.dirname(base_path)
    stem = os.path.splitext(os.path.basename(base_path))[0]
    candidates = []
    if os.path.isfile(base_path):
        candidates.append(base_path)
    if os.path.isdir(dir_path):
        prefix = f"{stem}_"
        for name in os.listdir(dir_path):
            if not name.endswith(".pt"):
                continue
            candidate_path = os.path.join(dir_path, name)
            candidate_stem = os.path.splitext(os.path.basename(candidate_path))[0]
            if candidate_stem == stem or candidate_stem.startswith(prefix):
                candidates.append(candidate_path)
    candidates = [path for path in dict.fromkeys(candidates) if os.path.isfile(path)]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (os.path.getmtime(path), path))
    return candidates[-1]


def _run_episode_done_step(policy_fn, max_episode_length: int) -> int:
    env = environ.CartPoleEnvironment(max_episode_length=max_episode_length, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        done_step = 0

        while done_step < max_episode_length:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done_step += 1
            if terminated or truncated:
                break

        return done_step
    finally:
        env.close()


def _load_actor_checkpoint_path(algo_type: str, actor_hidden_nn: np.ndarray):
    algo_upper = algo_type.upper()
    if algo_upper == "DQN":
        from Checkpointing import dqn_q_checkpoint_path
        return _latest_checkpoint_path(
            dqn_q_checkpoint_path(
                nn_hidden_layer_widths=actor_hidden_nn,
            ).file_path
        )
    from Checkpointing import pg_actor_checkpoint_path
    return _latest_checkpoint_path(
        pg_actor_checkpoint_path(
            algo_type=algo_type,
            actor_hidden_nn=actor_hidden_nn,
        ).file_path
    )


def _build_actor_checkpoint_model(algo_upper: str, actor_hidden_nn: np.ndarray):
    algo_upper = algo_upper.upper()
    if algo_upper in ("REINFORCE", "AC"):
        return Policy_NN(nn_hidden_layer_widths=actor_hidden_nn)
    if algo_upper in ("A2C", "PPO"):
        if algo_upper == "A2C":
            from A2C import PolicyNetwork
        else:
            from PPO import PolicyNetwork
        # Must match the activation used at training time. A2C/PPO now default to
        # tanh (the PPO-standard choice), so rebuild the eval network with tanh.
        return PolicyNetwork(4, 2, actor_hidden_nn, activation="tanh")
    if algo_upper == "DQN":
        from assignment2_repo.mylibrary import PolicyNetwork as DQN_QNetwork
        nn_depth = int(np.asarray(actor_hidden_nn).reshape(-1).size) + 2
        return DQN_QNetwork(nn_depth=nn_depth, nn_hidden_layer_widths=actor_hidden_nn)
    raise ValueError(f"Unsupported checkpoint-eval algorithm: {algo_upper}")


_ACTOR_CHECKPOINT_MODEL_CACHE: dict[tuple[str, str, tuple[int, ...]], torch.nn.Module] = {}


def _load_actor_checkpoint_model(algo_upper: str, checkpoint_path: str, actor_hidden_nn: np.ndarray):
    model = _build_actor_checkpoint_model(algo_upper, actor_hidden_nn)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model.eval()
    return model


def _get_cached_actor_checkpoint_model(algo_upper: str, checkpoint_path: str, actor_hidden_nn: np.ndarray):
    cache_key = (algo_upper.upper(), checkpoint_path, tuple(int(v) for v in np.asarray(actor_hidden_nn, dtype=np.int32).reshape(-1)))
    model = _ACTOR_CHECKPOINT_MODEL_CACHE.get(cache_key)
    if model is None:
        model = _load_actor_checkpoint_model(algo_upper, checkpoint_path, actor_hidden_nn)
        _ACTOR_CHECKPOINT_MODEL_CACHE[cache_key] = model
    return model


_VALID_POLICY_EVALUATION_METHODS = ("softmax", "argmax")


def _normalize_policy_evaluation_methods(policy_evaluation_method) -> list[str]:
    """Normalize the user-facing 'policy_evaluation_method' config value into a
    deduplicated list of supported method names ('softmax' / 'argmax').

    Accepts either a single string or any iterable of strings. None / empty
    falls back to ['softmax'] (the default used by other algorithms).
    """
    if policy_evaluation_method is None:
        return ["softmax"]
    if isinstance(policy_evaluation_method, str):
        candidates = [policy_evaluation_method]
    else:
        try:
            candidates = list(policy_evaluation_method)
        except TypeError:
            candidates = [policy_evaluation_method]
    if not candidates:
        return ["softmax"]
    seen: set[str] = set()
    deduped: list[str] = []
    for method in candidates:
        method_norm = str(method).strip().lower()
        if method_norm not in _VALID_POLICY_EVALUATION_METHODS:
            raise ValueError(
                "policy_evaluation_method entries must be one of "
                f"{_VALID_POLICY_EVALUATION_METHODS}, got {method!r}"
            )
        if method_norm not in seen:
            seen.add(method_norm)
            deduped.append(method_norm)
    return deduped


def _run_actor_checkpoint_episodes_batch(
    algo_upper: str,
    checkpoint_path: str,
    actor_hidden_nn: np.ndarray,
    max_eval_episode_length: int,
    policy_method: str,
    n_episodes_in_batch: int,
) -> np.ndarray:
    """Run a batch of independent evaluation episodes sequentially in one worker.

    Submitting one future per episode is fine for hundreds, but at 100k+ episodes
    the per-submit pickling overhead pins the main process and starves the worker
    queue. Batching collapses that overhead and keeps the cached model warm across
    the whole batch.
    """
    results = np.empty(n_episodes_in_batch, dtype=np.int32)
    for i in range(n_episodes_in_batch):
        results[i] = _run_actor_checkpoint_episode(
            algo_upper, checkpoint_path, actor_hidden_nn,
            max_eval_episode_length, policy_method,
        )
    return results


def _run_actor_checkpoint_episode(
    algo_upper: str,
    checkpoint_path: str,
    actor_hidden_nn: np.ndarray,
    max_eval_episode_length: int,
    policy_method: str = "argmax",
) -> int:
    model = _get_cached_actor_checkpoint_model(algo_upper, checkpoint_path, actor_hidden_nn)
    env = environ.CartPoleEnvironment(max_episode_length=max_eval_episode_length, render_mode=None)
    try:
        obs, _ = env.reset()
        done_step = 0
        while done_step < max_eval_episode_length:
            with torch.inference_mode():
                state = torch.as_tensor(obs, dtype=torch.float32)
                if algo_upper in ("REINFORCE", "AC"):
                    # Single-logit network: sigmoid(logit) is P(action=1).
                    logit = model(state)
                    prob_a1 = float(torch.sigmoid(logit).item())
                    if policy_method == "argmax":
                        action = int(prob_a1 >= 0.5)
                    else:  # softmax sample on Bernoulli(prob_a1)
                        action = int(np.random.random() < prob_a1)
                elif algo_upper == "DQN":
                    q_values = model(state)
                    if policy_method == "argmax":
                        action = int(torch.argmax(q_values, dim=-1).item())
                    else:
                        probs = torch.softmax(q_values, dim=-1).cpu().numpy().astype(np.float64)
                        probs = probs / probs.sum()
                        action = int(np.random.choice(probs.shape[-1], p=probs))
                else:  # A2C, PPO — 2-logit categorical policy
                    logits = model(state)
                    if policy_method == "argmax":
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                        probs = probs / probs.sum()
                        action = int(np.random.choice(probs.shape[-1], p=probs))
            obs, reward, terminated, truncated, _ = env.step(action)
            done_step += 1
            if terminated or truncated:
                break
        return done_step
    finally:
        env.close()


def run_actor_checkpoint_evaluation(
    *,
    included_algo_checkpoint_eval,
    max_eval_episode_length,
    separate_algorithm_plots,
    show_curve_plots,
    unused_cpu_cores=0,
    policy_evaluation_method=None,
):
    cfg = included_algo_checkpoint_eval or {}
    enabled_algos = [
        algo.upper()
        for algo in ("REINFORCE", "AC", "A2C", "DQN", "PPO")
        if bool((cfg.get(algo.upper()) or {}).get("enabled", False))
    ]
    if not enabled_algos:
        return

    policy_methods = _normalize_policy_evaluation_methods(policy_evaluation_method)
    multi_methods = len(policy_methods) > 1

    n_episodes = int(cfg.get("n_episodes", 1000))
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    algo_jobs = []
    for algo_upper in enabled_algos:
        algo_cfg = cfg.get(algo_upper) or {}
        hidden_widths_key = "nn_hidden_layer_widths" if algo_upper == "DQN" else "actor_hidden_nn"
        actor_hidden_nn = np.asarray(
            algo_cfg.get(hidden_widths_key, algo_cfg.get("actor_hidden_nn")),
            dtype=np.int32,
        )
        checkpoint_path = _load_actor_checkpoint_path(algo_upper, actor_hidden_nn)
        if checkpoint_path is None:
            ckpt_label = "Q-network" if algo_upper == "DQN" else "actor"
            print(f"[{algo_upper}] No {ckpt_label} checkpoint found on disk; skipping.")
            continue
        ckpt_label = "Q-network" if algo_upper == "DQN" else "actor"
        print(f"[{algo_upper}] Loading {ckpt_label} checkpoint: {checkpoint_path}")

        for method in policy_methods:
            series_key = f"{algo_upper}_{method}" if multi_methods else algo_upper
            display_label = f"{algo_upper} ({method})"
            algo_jobs.append(
                {
                    "algo_upper": algo_upper,
                    "actor_hidden_nn": actor_hidden_nn,
                    "checkpoint_path": checkpoint_path,
                    "policy_method": method,
                    "series_key": series_key,
                    "display_label": display_label,
                }
            )

    if not algo_jobs:
        return

    total_tasks = len(algo_jobs) * n_episodes
    cpu_count = os.cpu_count() or 1
    if unused_cpu_cores is None:
        unused_cpu_cores = 0
    unused_cpu_cores = int(unused_cpu_cores)
    if unused_cpu_cores < 0:
        unused_cpu_cores = 0
    available_cpus = max(1, cpu_count - unused_cpu_cores)
    max_workers = min(total_tasks, available_cpus)

    print(
        f"CPU cores available: {cpu_count} "
        f"(reserving UNUSED_CPU_CORES={unused_cpu_cores} => using {available_cpus}). "
        f"Total tasks: {total_tasks} "
        f"({len(algo_jobs)} algorithm(s) × {n_episodes} episode(s)). "
        f"Parallel workers: {max_workers}.\n"
    )

    if separate_algorithm_plots:
        figure_map = {
            job["algo_upper"]: plt.subplots(figsize=(10, 6))
            for job in algo_jobs
        }
    else:
        fig, axis = plt.subplots(figsize=(12, 7))
        figure_map = {"combined": (fig, axis)}

    results_by_series = {
        job["series_key"]: np.empty(n_episodes, dtype=np.int32)
        for job in algo_jobs
    }
    series_to_label = {job["series_key"]: job["display_label"] for job in algo_jobs}
    series_to_algo = {job["series_key"]: job["algo_upper"] for job in algo_jobs}
    eval_numbers = np.arange(1, n_episodes + 1, dtype=np.int32)
    pbars = {
        job["series_key"]: _create_step_progress_bar(
            total=n_episodes,
            desc=f"{job['algo_upper']} ({job.get('policy_method') or 'argmax'}) checkpoint eval",
            position=index,
            leave=True,
        )
        for index, job in enumerate(algo_jobs)
    }

    # Submit per-algo work in chunks so the main process isn't stuck pickling
    # hundreds of thousands of individual submits. Target ~200 chunks per algo to
    # keep progress bars responsive while bounding submission overhead.
    target_chunks_per_algo = max(200, 4 * max_workers)
    target_chunks_per_algo = min(target_chunks_per_algo, n_episodes)
    chunk_size = max(1, (n_episodes + target_chunks_per_algo - 1) // target_chunks_per_algo)

    future_meta = {}
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for job in algo_jobs:
            for episode_start in range(0, n_episodes, chunk_size):
                episode_count = min(chunk_size, n_episodes - episode_start)
                future = executor.submit(
                    _run_actor_checkpoint_episodes_batch,
                    job["algo_upper"],
                    job["checkpoint_path"],
                    job["actor_hidden_nn"],
                    max_eval_episode_length,
                    job.get("policy_method") or "softmax",
                    episode_count,
                )
                future_meta[future] = (job["series_key"], episode_start, episode_count)
                futures.append(future)

        try:
            for future in as_completed(futures):
                series_key, episode_start, episode_count = future_meta[future]
                batch_results = future.result()
                results_by_series[series_key][episode_start:episode_start + episode_count] = batch_results
                pbars[series_key].set_postfix_str(f"done_step={int(batch_results[-1])}", refresh=False)
                pbars[series_key].update(int(episode_count))
        finally:
            for pbar in pbars.values():
                pbar.close()
            print()

    algo_line_styles = {
        "REINFORCE": {"color": "#d62728", "marker": "o", "ls": "-"},
        "AC": {"color": "#1f77b4", "marker": "s", "ls": "--"},
        "A2C": {"color": "#2ca02c", "marker": "^", "ls": "-."},
        "DQN": {"color": "#ff7f0e", "marker": "x", "ls": (0, (3, 1, 1, 1))},
        "PPO": {"color": "#9467bd", "marker": "D", "ls": ":"},
    }
    method_linestyle_overrides = {
        "softmax": "-",
        "argmax": (0, (4, 2)),
    }
    series_to_method = {job["series_key"]: job["policy_method"] for job in algo_jobs}

    for series_key, done_steps in results_by_series.items():
        algo_upper = series_to_algo[series_key]
        plot_key = algo_upper if separate_algorithm_plots else "combined"
        fig, axis = figure_map[plot_key]
        style = dict(algo_line_styles.get(algo_upper, {}))
        if multi_methods:
            method_for_series = series_to_method.get(series_key)
            if method_for_series in method_linestyle_overrides:
                style["ls"] = method_linestyle_overrides[method_for_series]
        axis.plot(
            eval_numbers,
            done_steps,
            marker=style.get("marker", "o"),
            linestyle=style.get("ls", "-"),
            linewidth=2.0,
            color=style.get("color", None),
            label=series_to_label[series_key],
            zorder=5,
        )
        axis.set_title(
            f"{algo_upper} checkpoint evaluation" if separate_algorithm_plots else "Checkpoint evaluation"
        )
        axis.set_xlabel("evaluation number")
        axis.set_ylabel("done step")
        axis.grid(True, alpha=0.25)

    if separate_algorithm_plots:
        for algo_upper, (fig, axis) in figure_map.items():
            if axis.lines:
                axis.legend()
                fig.tight_layout()
                output_path = os.path.join(plots_dir, f"{algo_upper}_checkpoint_evaluation.png")
                fig.savefig(output_path, dpi=300)
                print(f"[{algo_upper}] Saved checkpoint evaluation plot to {output_path}")
    else:
        fig, axis = figure_map["combined"]
        if axis.lines:
            axis.legend()
            fig.tight_layout()
            output_path = os.path.join(plots_dir, "checkpoint_evaluation.png")
            fig.savefig(output_path, dpi=300)
            print(f"Saved checkpoint evaluation plot to {output_path}")

    if show_curve_plots:
        plt.show()


def _build_checkpoint_eval_summary(
    *,
    results_by_series,
    series_to_label,
    series_to_algo,
    series_to_method=None,
    multi_methods=False,
    output_dir,
    max_eval_episode_length=None,
    curves_confidence_interval=None,
    n_episodes=None,
    title_suffix="",
):
    """Emit md (LaTeX) / csv / boxplot / mean-std summary of per-episode done_steps.

    One row per series: in single-method runs that's one row per algorithm
    (e.g. "DQN"); in multi-method runs it's one row per (algo, method) pair
    (e.g. "DQN (softmax)", "DQN (argmax)"). All files are suffixed with
    RUN_TIMESTAMP for traceability across runs.
    """
    import csv as _csv

    os.makedirs(output_dir, exist_ok=True)
    series_to_method = series_to_method or {}

    rows_data = []
    for series_key, done_steps in results_by_series.items():
        values = np.asarray(done_steps, dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        algo = series_to_algo.get(series_key, series_key)
        method = series_to_method.get(series_key)
        if multi_methods and method is not None:
            row_label = f"{algo} ({method})"
        else:
            row_label = algo
        q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
        rows_data.append({
            "algorithm": row_label,
            "algo_upper": algo,
            "policy_method": method,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "q1": q1,
            "median": median,
            "q3": q3,
            "n": int(values.size),
            "values": values,
        })

    headers = ["Algorithm", "# Observations", "Mean", "Std", "Q1", "Median", "Q3"]
    n_obs_str = f"{int(n_episodes):,}" if n_episodes is not None else ""

    def _fmt_num(value, decimals=2):
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{value:,.{decimals}f}"

    text_rows = []
    csv_rows = []
    for row in rows_data:
        text_rows.append([
            row["algorithm"],
            n_obs_str,
            _fmt_num(row["mean"]),
            _fmt_num(row["std"]),
            _fmt_num(row["q1"]),
            _fmt_num(row["median"]),
            _fmt_num(row["q3"]),
        ])
        csv_rows.append([
            row["algorithm"],
            str(int(n_episodes)) if n_episodes is not None else "",
            f"{row['mean']:.6f}" if np.isfinite(row["mean"]) else "",
            f"{row['std']:.6f}" if np.isfinite(row["std"]) else "",
            f"{row['q1']:.6f}" if np.isfinite(row["q1"]) else "",
            f"{row['median']:.6f}" if np.isfinite(row["median"]) else "",
            f"{row['q3']:.6f}" if np.isfinite(row["q3"]) else "",
        ])

    if text_rows:
        widths = [len(h) for h in headers]
        for r in text_rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))

        def _fmt_row(cells):
            out = []
            for i, cell in enumerate(cells):
                if i == 0:
                    out.append(cell.ljust(widths[i]))
                else:
                    out.append(cell.rjust(widths[i]))
            return "  ".join(out)

        sep = "  ".join("-" * w for w in widths)
        rendered_lines = [_fmt_row(headers), sep] + [_fmt_row(r) for r in text_rows]
        rendered_text = "\n".join(rendered_lines)
    else:
        rendered_text = "(no completed series to summarize)"

    truncation_int = int(max_eval_episode_length) if max_eval_episode_length is not None else 0
    truncation_caption = f"{truncation_int:,}"
    title_stdout = f"Checkpoint evaluation summary - mean/std/quartiles per series (truncation={truncation_caption})"

    def _safe_print(s):
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode("ascii"))

    _safe_print("\n" + "=" * len(title_stdout))
    _safe_print(title_stdout)
    _safe_print("=" * len(title_stdout))
    _safe_print(rendered_text)
    _safe_print("")

    # ── LaTeX table template (saved as .txt so renderers don't try to parse it) ──
    md_path = os.path.join(output_dir, f"checkpoint_eval_summary_{RUN_TIMESTAMP}.txt")
    if text_rows:
        algo_width = max(len(row["algorithm"]) for row in rows_data)
        obs_width = max(len(n_obs_str), len("# Observations"))
        num_keys = ("mean", "std", "q1", "median", "q3")
        num_strs = {k: [_fmt_num(row[k]) for row in rows_data] for k in num_keys}
        num_widths = {k: max(len(s) for s in num_strs[k]) for k in num_keys}
        body_lines = []
        for i, row in enumerate(rows_data):
            cells = [
                row["algorithm"].ljust(algo_width),
                n_obs_str.rjust(obs_width),
                num_strs["mean"][i].rjust(num_widths["mean"]),
                num_strs["std"][i].rjust(num_widths["std"]),
                num_strs["q1"][i].rjust(num_widths["q1"]),
                num_strs["median"][i].rjust(num_widths["median"]),
                num_strs["q3"][i].rjust(num_widths["q3"]),
            ]
            body_lines.append(" & ".join(cells) + r" \\")
        body = "\n".join(body_lines)
    else:
        body = "% (no completed series to summarize)"

    latex_lines = [
        r"\begin{table}[t]",
        r"\vskip 0.15in",
        r"\begin{center}",
        r"\begin{small}",
        r"\begin{sc}",
        f"\\caption{{Statistical factors (truncation={truncation_caption})}}",
        f"\\label{{tab:results_summary_{truncation_int}}}",
        r"\begin{tabular}{l r r r r r r }",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{\# Observations}& \textbf{Mean}& \textbf{Std}& \textbf{Q1}& \textbf{Median}& \textbf{Q3}\\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{sc}",
        r"\end{small}",
        r"\end{center}",
        r"\vskip -0.1in",
        r"\end{table}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")

    # ── CSV ──
    csv_path = os.path.join(output_dir, f"checkpoint_eval_summary_{RUN_TIMESTAMP}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(headers)
        for r in csv_rows:
            writer.writerow(r)

    boxplot_path = None
    mean_ci_plot_path = None

    if rows_data:
        try:
            valid_algorithms = [row["algorithm"] for row in rows_data]
            all_data = [row["values"].tolist() for row in rows_data]

            fig, ax = plt.subplots(figsize=(max(10, 2.0 * len(valid_algorithms)), 6))
            positions = np.arange(1, len(valid_algorithms) + 1, dtype=float)
            ax.boxplot(
                all_data,
                positions=positions,
                widths=0.5,
                vert=True,
                patch_artist=True,
                showmeans=True,
                boxprops=dict(facecolor="#d9d9d9", edgecolor="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=1.0),
                capprops=dict(color="black", linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.4),
                meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=4),
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(valid_algorithms)
            ax.set_ylabel("Done step")
            ax.set_title(
                f"Checkpoint evaluation box plot (truncation={truncation_caption}){title_suffix}",
                multialignment="center",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            boxplot_path = os.path.join(output_dir, f"checkpoint_eval_summary_boxplot_{RUN_TIMESTAMP}.png")
            fig.savefig(boxplot_path, dpi=300)
            plt.close(fig)
            print(f"Saved checkpoint evaluation box plot to: {boxplot_path}")

            # Confidence band: when curves_confidence_interval is None we keep
            # the original mean ± 1σ shading (≈ 68.27% under normality) and
            # leave the CI line out of the title; otherwise we scale the band
            # by the z-score matching the requested central probability.
            if curves_confidence_interval is None:
                ci_scale = 1.0
                ci_title_suffix = ""
            else:
                from scipy.stats import norm as _norm
                ci_fraction = float(curves_confidence_interval) / 100.0
                ci_fraction = min(max(ci_fraction, 0.0), 0.999999)
                ci_scale = float(_norm.ppf((1.0 + ci_fraction) / 2.0))
                ci_title_suffix = f"\nCI: {curves_confidence_interval}%"

            fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(rows_data)), 6))
            x_positions = np.arange(1, len(rows_data) + 1, dtype=float)
            half_width = 0.35
            palette = plt.get_cmap("tab10")
            for idx, (x_pos, row) in enumerate(zip(x_positions, rows_data)):
                mean_value = float(row["mean"])
                std_value = float(row["std"])
                if not np.isfinite(mean_value):
                    continue
                margin = ci_scale * (std_value if np.isfinite(std_value) else 0.0)
                lower = mean_value - margin
                upper = mean_value + margin
                color = palette(idx % palette.N)
                ax.fill_between(
                    [x_pos - half_width, x_pos + half_width],
                    [lower, lower],
                    [upper, upper],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
                ax.plot([x_pos - half_width, x_pos + half_width], [mean_value, mean_value], color=color, linewidth=1.8, label=row["algorithm"])

            ax.set_xticks(x_positions)
            ax.set_xticklabels([row["algorithm"] for row in rows_data])
            ax.set_ylabel("Done step")
            ax.set_title(
                f"Checkpoint evaluation mean ± std (truncation={truncation_caption}){ci_title_suffix}{title_suffix}",
                multialignment="center",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            mean_ci_plot_path = os.path.join(output_dir, f"checkpoint_eval_summary_mean_std_{RUN_TIMESTAMP}.png")
            fig.savefig(mean_ci_plot_path, dpi=300)
            plt.close(fig)
            print(f"Saved checkpoint evaluation mean/std plot to: {mean_ci_plot_path}")
        except Exception as exc:
            print(f"[summary] Failed to render checkpoint evaluation summary plots: {exc}")

    print(f"Saved checkpoint evaluation summary to: {md_path}")
    print(f"Saved checkpoint evaluation summary to: {csv_path}")
    print()

    return {
        "rows": rows_data,
        "headers": headers,
        "latex_path": md_path,
        "csv_path": csv_path,
        "boxplot_path": boxplot_path,
        "mean_std_plot_path": mean_ci_plot_path,
        "text": rendered_text,
    }


def run_actor_checkpoint_evaluation_exhaustive(
    *,
    included_algo_checkpoint_eval,
    max_eval_episode_length,
    n_episodes,
    plot_smoothing_window=None,
    show_curve_smoothing_windows=None,
    separate_algorithm_plots=False,
    show_curve_plots=False,
    unused_cpu_cores=0,
    policy_evaluation_method=None,
    checkpoint_evaluation_plots=True,
    checkpoint_evaluation_analysis=True,
    curves_confidence_interval=None,
    plot_parameters=None,
):
    start_time = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Experiment started at: {start_human}\n")
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Start the process at: {start_human}\n")
    cfg = included_algo_checkpoint_eval or {}
    enabled_algos = [
        algo.upper()
        for algo in ("REINFORCE", "AC", "A2C", "DQN", "PPO")
        if bool((cfg.get(algo.upper()) or {}).get("enabled", False))
    ]
    if not enabled_algos:
        return

    policy_methods = _normalize_policy_evaluation_methods(policy_evaluation_method)
    multi_methods = len(policy_methods) > 1

    plot_smoothing_windows = np.atleast_1d(
        np.asarray(plot_smoothing_window if plot_smoothing_window is not None else np.array([1]), dtype=np.int32)
    )
    if plot_smoothing_windows.size < 1:
        raise ValueError("plot_smoothing_window must contain at least one value.")

    show_curve_smoothing_windows = np.atleast_1d(
        np.asarray(
            show_curve_smoothing_windows if show_curve_smoothing_windows is not None else np.array([101, 201]),
            dtype=np.int32,
        )
    )
    if show_curve_smoothing_windows.size < 1:
        raise ValueError("show_curve_smoothing_windows must contain at least one value.")

    n_episodes = int(n_episodes)
    output_dir = os.path.abspath("Checkpoint Evaluation Trials")
    os.makedirs(output_dir, exist_ok=True)

    # Optional centered title suffix: "\n#episodes: <n>" when enabled via
    # plot_parameters; empty string otherwise (so titles look exactly as before).
    plot_parameters = plot_parameters or {}
    show_n_episodes_in_title = bool(plot_parameters.get("n_episodes", False))
    title_suffix = f"\n#episodes: {n_episodes:,}" if show_n_episodes_in_title else ""

    algo_jobs = []
    for algo_upper in enabled_algos:
        algo_cfg = cfg.get(algo_upper) or {}
        hidden_widths_key = "nn_hidden_layer_widths" if algo_upper == "DQN" else "actor_hidden_nn"
        actor_hidden_nn = np.asarray(
            algo_cfg.get(hidden_widths_key, algo_cfg.get("actor_hidden_nn")),
            dtype=np.int32,
        )
        checkpoint_path = _load_actor_checkpoint_path(algo_upper, actor_hidden_nn)
        if checkpoint_path is None:
            ckpt_label = "Q-network" if algo_upper == "DQN" else "actor"
            print(f"[{algo_upper}] No {ckpt_label} checkpoint found on disk; skipping.")
            continue
        ckpt_label = "Q-network" if algo_upper == "DQN" else "actor"
        print(f"[{algo_upper}] Loading {ckpt_label} checkpoint: {checkpoint_path}")

        for method in policy_methods:
            series_key = f"{algo_upper}_{method}" if multi_methods else algo_upper
            display_label = f"{algo_upper} ({method})"
            algo_jobs.append(
                {
                    "algo_upper": algo_upper,
                    "actor_hidden_nn": actor_hidden_nn,
                    "checkpoint_path": checkpoint_path,
                    "policy_method": method,
                    "series_key": series_key,
                    "display_label": display_label,
                }
            )

    if not algo_jobs:
        return

    cpu_count = os.cpu_count() or 1
    if unused_cpu_cores is None:
        unused_cpu_cores = 0
    unused_cpu_cores = int(unused_cpu_cores)
    if unused_cpu_cores < 0:
        unused_cpu_cores = 0
    available_cpus = max(1, cpu_count - unused_cpu_cores)
    max_workers = min(len(algo_jobs) * max(1, min(n_episodes, available_cpus)), available_cpus)

    print(f"Running checkpoint evaluation: {len(algo_jobs)} algorithm(s) × {n_episodes} episode(s).\n")

    results_by_series = {
        job["series_key"]: np.empty(n_episodes, dtype=np.int32)
        for job in algo_jobs
    }
    series_to_label = {job["series_key"]: job["display_label"] for job in algo_jobs}
    series_to_algo = {job["series_key"]: job["algo_upper"] for job in algo_jobs}
    eval_numbers = np.arange(1, n_episodes + 1, dtype=np.int32)
    pbars = {
        job["series_key"]: _create_step_progress_bar(
            total=n_episodes,
            desc=(
                f"{job['algo_upper']} ({job['policy_method']})"
                if multi_methods else job["algo_upper"]
            ),
            position=index,
            leave=True,
            mininterval=0.5,
            unit="ep",
        )
        for index, job in enumerate(algo_jobs)
    }

    # Chunk episodes per algo. The submission count drives both pickle overhead and
    # how often the progress bars redraw: too many chunks pins the main process and
    # spams the terminal, too few starves workers. Aim for a small multiple of
    # max_workers spread across all algos, giving each algo ~20 progress updates.
    target_total_chunks = max(2 * max_workers, len(algo_jobs))
    target_per_algo = max(20, target_total_chunks // max(len(algo_jobs), 1))
    target_per_algo = min(target_per_algo, n_episodes)
    chunk_size = max(1, (n_episodes + target_per_algo - 1) // target_per_algo)

    future_meta = {}
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for job in algo_jobs:
            for episode_start in range(0, n_episodes, chunk_size):
                episode_count = min(chunk_size, n_episodes - episode_start)
                future = executor.submit(
                    _run_actor_checkpoint_episodes_batch,
                    job["algo_upper"],
                    job["checkpoint_path"],
                    job["actor_hidden_nn"],
                    max_eval_episode_length,
                    job.get("policy_method") or "softmax",
                    episode_count,
                )
                future_meta[future] = (job["series_key"], episode_start, episode_count)
                futures.append(future)

        try:
            for future in as_completed(futures):
                series_key, episode_start, episode_count = future_meta[future]
                batch_results = future.result()
                results_by_series[series_key][episode_start:episode_start + episode_count] = batch_results
                pbars[series_key].update(int(episode_count))
        finally:
            for pbar in pbars.values():
                pbar.close()
            print()

    plot_filename_tag = "-".join(enabled_algos)
    series_to_method = {job["series_key"]: job["policy_method"] for job in algo_jobs}

    # When multiple policy methods are active, render a separate set of curve
    # plots per method (so each plot contains one curve per algo, all using the
    # same evaluation policy). Single-method runs keep one combined plot.
    method_groups = policy_methods if multi_methods else [None]

    def _curve_label(series_key: str, method_filter: str | None) -> str:
        algo_upper = series_to_algo[series_key]
        if method_filter is None and multi_methods:
            return series_to_label[series_key]
        return algo_upper

    plot_configs = []
    if separate_algorithm_plots:
        for algo_upper in enabled_algos:
            for method_filter in method_groups:
                for window in show_curve_smoothing_windows:
                    window = int(window)
                    suffix_label = "not smoothed plot" if window <= 1 else "smoothed plot"
                    title_prefix = (
                        f"{algo_upper} ({method_filter})" if method_filter is not None else algo_upper
                    )
                    plot_configs.append(
                        {
                            "algo_upper": algo_upper,
                            "policy_method": method_filter,
                            "window": window,
                            "plot": LearningCurvePlot(title=f"{title_prefix} checkpoint evaluation - {suffix_label}{title_suffix}"),
                        }
                    )
    else:
        for method_filter in method_groups:
            for window in show_curve_smoothing_windows:
                window = int(window)
                suffix_label = "not smoothed plot" if window <= 1 else "smoothed plot"
                title_prefix = (
                    f"Checkpoint evaluation ({method_filter})" if method_filter is not None else "Checkpoint evaluation"
                )
                plot_configs.append(
                    {
                        "policy_method": method_filter,
                        "window": window,
                        "plot": LearningCurvePlot(title=f"{title_prefix} - {suffix_label}{title_suffix}"),
                    }
                )

    algo_color_map = {
        "REINFORCE": "#d62728",
        "AC": "#1f77b4",
        "A2C": "#2ca02c",
        "DQN": "#ff7f0e",
        "PPO": "#9467bd",
    }

    for pc in plot_configs:
        method_filter = pc.get("policy_method")
        algo_filter = pc.get("algo_upper") if separate_algorithm_plots else None
        for series_key, done_steps in results_by_series.items():
            if method_filter is not None and series_to_method.get(series_key) != method_filter:
                continue
            algo_upper = series_to_algo[series_key]
            if algo_filter is not None and algo_upper != algo_filter:
                continue
            smoothed_steps = _apply_optional_smoothing(np.asarray(done_steps, dtype=np.float32), int(pc["window"]))
            smoothed_steps = np.minimum(smoothed_steps, float(max_eval_episode_length))
            pc["plot"].add_curve(
                eval_numbers,
                smoothed_steps,
                label=_curve_label(series_key, method_filter),
                ls="-",
                color=algo_color_map.get(algo_upper),
            )

    if checkpoint_evaluation_plots:
        for pc in plot_configs:
            window = int(pc["window"])
            suffix = f"w{window}-not-smoothed" if window <= 1 else f"w{window}-smoothed"
            method_filter = pc.get("policy_method")
            name_parts = [plot_filename_tag]
            if separate_algorithm_plots:
                name_parts = [pc["algo_upper"]]
            if method_filter is not None:
                name_parts.append(method_filter)
            name_parts.append(suffix)
            filename = "_".join(name_parts) + ".png"
            output_path = os.path.abspath(os.path.join(output_dir, filename))
            pc["plot"].add_hline(max_eval_episode_length, label="CartPole Optimum")
            saved_path = pc["plot"].save(output_path)
            print(f"Saved checkpoint evaluation plot to {saved_path}")

        if not separate_algorithm_plots and len(show_curve_smoothing_windows) == 2:
            desired_windows = [int(show_curve_smoothing_windows[0]), int(show_curve_smoothing_windows[1])]
            for method_filter in method_groups:
                combined_fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                suptitle_method = f" [{method_filter}]" if method_filter is not None else ""
                combined_fig.suptitle(
                    f"Twin_{plot_filename_tag}{suptitle_method} (w={desired_windows[0]} and w={desired_windows[1]}){title_suffix}",
                    multialignment="center",
                )
                for ax, w in zip(axes, desired_windows):
                    for series_key, done_steps in results_by_series.items():
                        if method_filter is not None and series_to_method.get(series_key) != method_filter:
                            continue
                        algo_upper = series_to_algo[series_key]
                        smoothed_steps = _apply_optional_smoothing(np.asarray(done_steps, dtype=np.float32), int(w))
                        smoothed_steps = np.minimum(smoothed_steps, float(max_eval_episode_length))
                        ax.plot(
                            eval_numbers,
                            smoothed_steps,
                            label=_curve_label(series_key, method_filter),
                            color=algo_color_map.get(algo_upper),
                            linestyle="-",
                            linewidth=2.0,
                            zorder=5,
                        )
                    ax.axhline(max_eval_episode_length, color="gray", linestyle=":", linewidth=1.5, label="CartPole Optimum")
                    sub_title = f"Checkpoint evaluation - {'not smoothed plot' if w <= 1 else 'smoothed plot'}"
                    if method_filter is not None:
                        sub_title = f"Checkpoint evaluation ({method_filter}) - {'not smoothed plot' if w <= 1 else 'smoothed plot'}"
                    ax.set_title(sub_title)
                    ax.set_xlabel("evaluation number")
                    ax.set_ylabel("done step")
                    ax.grid(True, alpha=0.25)
                    ax.legend()
                try:
                    combined_fig.tight_layout()
                    twin_parts = [plot_filename_tag]
                    if method_filter is not None:
                        twin_parts.append(method_filter)
                    twin_parts.append(f"twin_w{desired_windows[0]}-w{desired_windows[1]}_{RUN_TIMESTAMP}")
                    twin_filename = "_".join(twin_parts) + ".png"
                    twin_output_path = os.path.abspath(os.path.join(output_dir, twin_filename))
                    combined_fig.savefig(twin_output_path, dpi=300)
                    print(f"Saved checkpoint evaluation twin plot to {twin_output_path}")
                    if show_curve_plots:
                        plt.show(block=False)
                except Exception as exc:
                    print(f"[plot] Failed to create combined subplot preview: {exc}")

    if checkpoint_evaluation_analysis:
        try:
            _build_checkpoint_eval_summary(
                results_by_series=results_by_series,
                series_to_label=series_to_label,
                series_to_algo=series_to_algo,
                series_to_method=series_to_method,
                multi_methods=multi_methods,
                output_dir=output_dir,
                max_eval_episode_length=max_eval_episode_length,
                curves_confidence_interval=curves_confidence_interval,
                n_episodes=n_episodes,
                title_suffix=title_suffix,
            )
        except Exception as exc:
            print(f"[summary] Failed to build checkpoint evaluation summary: {exc}")

    total_time = (time.perf_counter() - start_time) / 60.0
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Total execution time: {total_time:.3f} minutes\n")
    print(f"\nExperiment finished in {total_time:.3f} minutes.")

    if checkpoint_evaluation_plots and show_curve_plots:
        plt.show(block=True)

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

################[ Shared engineering-trick helpers (init & observation normalization) ]################
# These small helpers are reused by every network class in the project (Policy_NN /
# Value_NN here, and the PolicyNetwork / ValueNetwork classes in A2C.py / PPO.py) so
# that the same orthogonal-initialization and running observation-normalization
# behaviour is available to all four policy-gradient algorithms.

def init_mlp_orthogonal(net, head_gain, hidden_gain=None):
    """Orthogonally initialize every nn.Linear in `net`.

    Hidden layers use `hidden_gain` (default sqrt(2), the standard choice for
    tanh/ReLU MLPs); the final (output) layer uses the smaller `head_gain`
    (typically 0.01 for a policy head and 1.0 for a value head) so the network
    starts close to a uniform policy / unbiased value. Biases are zeroed.
    This is one of the most consistently beneficial PPO/A2C implementation
    details (Engstrom et al., 2020 — "Implementation Matters").
    """
    if hidden_gain is None:
        hidden_gain = float(np.sqrt(2))
    linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for index, layer in enumerate(linear_layers):
        gain = head_gain if index == len(linear_layers) - 1 else hidden_gain
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def register_obs_norm(module, n_features=4):
    """Register running-statistics buffers for observation normalization.

    Buffers are only created when normalization is actually requested, so the
    state_dict of a network trained WITHOUT obs-normalization keeps exactly the
    same keys as before — existing checkpoints continue to load with strict=True.
    """
    module.register_buffer("obs_running_mean", torch.zeros(n_features))
    module.register_buffer("obs_running_var", torch.ones(n_features))
    module.register_buffer("obs_norm_count", torch.tensor(1e-4))


def normalize_obs(module, x):
    """Apply (and clamp) observation normalization if the module has it enabled."""
    if not getattr(module, "normalize_obs", False):
        return x
    mean = module.obs_running_mean
    var = module.obs_running_var
    return torch.clamp((x - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)


@torch.no_grad()
def update_obs_norm(module, batch):
    """Welford parallel-update of the running observation mean/variance.

    Call once per `update()` with the batch of states. No-op when the module
    was built with normalize_obs=False.
    """
    if not getattr(module, "normalize_obs", False):
        return
    batch = torch.as_tensor(batch, dtype=torch.float32).reshape(-1, module.obs_running_mean.shape[-1])
    if batch.shape[0] == 0:
        return
    batch_mean = batch.mean(0)
    batch_var = batch.var(0, unbiased=False)
    batch_count = float(batch.shape[0])
    delta = batch_mean - module.obs_running_mean
    total = module.obs_norm_count + batch_count
    module.obs_running_mean.add_(delta * batch_count / total)
    m_a = module.obs_running_var * module.obs_norm_count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.pow(2) * module.obs_norm_count * batch_count / total
    module.obs_running_var.copy_(m2 / total)
    module.obs_norm_count.copy_(total)


def resolve_activation(activation):
    """Map an activation name (or class) to an nn activation class.

    Accepts the strings "tanh"/"relu" (case-insensitive) or an nn.Module
    subclass directly, so callers can pass either a configurable name or a class.
    """
    if isinstance(activation, str):
        key = activation.strip().lower()
        if key == "tanh":
            return nn.Tanh
        if key == "relu":
            return nn.ReLU
        raise ValueError(f"Unsupported activation name: {activation!r}")
    return activation
#########################################################

################[ Value_NN (Critic) ]################
class Value_NN(nn.Module):      # This is neural network for the critic (Values NN), which can be used as either a state-value critic V_phi(s) or an action-value critic Q_phi(s) depending on the output size and how it's trained.
    """State-value critic V_phi(s), or Q-value critic Q_phi(s) when used per-action."""
    def __init__(self, nn_hidden_layer_widths=np.array([64, 64]), output_size=1,
                 orthogonal_init=True, normalize_obs=False):
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

        # Engineering tricks (default-on init; obs-norm opt-in for checkpoint compat)
        self.normalize_obs = bool(normalize_obs)
        if self.normalize_obs:
            register_obs_norm(self, n_features=4)
        if orthogonal_init:
            init_mlp_orthogonal(self.net, head_gain=1.0)  # value head gain = 1.0

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        return self.net(normalize_obs(self, state))
#########################################################

################[ Policy_NN Class              ]################
class Policy_NN(nn.Module):
    def __init__(self, nn_hidden_layer_widths=np.array([5]),
                 orthogonal_init=True, normalize_obs=False):     # nn_depth is the total number of layers (input + hidden + output), and nn_hidden_layer_widths are the input and output sizes of all layers in depth order (the first number is for the input and the last number is for the output). For example, if nn_depth=3 and nn_hidden_layer_widths=[5, 5], then the network will have an input layer of size 4 (the state dimension), a hidden layer of size 5, another hidden layer of size 5, and an output layer of size 1 (the action dimension).
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

        # Engineering tricks (default-on init; obs-norm opt-in for checkpoint compat)
        self.normalize_obs = bool(normalize_obs)
        if self.normalize_obs:
            register_obs_norm(self, n_features=4)
        if orthogonal_init:
            init_mlp_orthogonal(self.net, head_gain=0.01)  # policy head gain = 0.01

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        return self.net(normalize_obs(self, state))
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
    # Accept both the legacy flat key and the new nested 'checkpoints' dict form:
    #   "checkpoints": {
    #       "use_saved_disk_networks_checkpoints": bool,
    #       "skip_selection_hyperparameter_match": bool,
    #   }
    # When the nested form is used, mirror the bool back into the flat key so
    # downstream legend/label code that references it by name keeps working.
    _ckpt_cfg = gc.get("checkpoints")
    if isinstance(_ckpt_cfg, dict):
        use_saved_disk_networks_checkpoints = bool(_ckpt_cfg.get("use_saved_disk_networks_checkpoints", False))
        skip_selection_hyperparameter_match = bool(_ckpt_cfg.get("skip_selection_hyperparameter_match", False))
        gc.setdefault("use_saved_disk_networks_checkpoints", use_saved_disk_networks_checkpoints)
    else:
        use_saved_disk_networks_checkpoints = bool(gc.get("use_saved_disk_networks_checkpoints", False))
        skip_selection_hyperparameter_match = False
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
                global_config=gc,
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
                global_config=gc,
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
                global_config=gc,
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
                global_config=gc,
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
                global_config=gc,
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
    title_parameters = gc.get("title_parameters", {})

    def _format_title_value(value):
        if isinstance(value, (bool, np.bool_)):
            return "✓" if bool(value) else "✗"
        return str(value)

    def _build_title_suffix() -> str:
        if not isinstance(title_parameters, dict):
            return ""
        parts = []
        for key, entry in title_parameters.items():
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            label, show = entry
            if not show:
                continue
            if key not in gc:
                continue
            parts.append(f"{label}{_format_title_value(gc[key])}")
        return ", ".join(parts)

    title_suffix = _build_title_suffix()

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
            subtitle_str = title_suffix if title_suffix else None
            # In separate mode, defer figure creation to _finalize_algo_plot so that
            # not-yet-populated algo windows don't pop up as empty placeholders the
            # moment another algo's plot is shown non-blocking.
            plot_obj = LearningCurvePlot(title=title_str, subtitle=subtitle_str) if create_figures else None
            cfgs.append({
                "window": window,
                "is_not_smoothed": is_not_smoothed,
                "title": title_str,
                "subtitle": subtitle_str,
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
    plots_dir = "Trial Continuation Analysis" if use_saved_disk_networks_checkpoints else "plots"
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
                pc["plot"] = LearningCurvePlot(title=pc["title"], subtitle=pc.get("subtitle"))

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
        # Emitted before the worker pool starts so the messages are not
        # interleaved with tqdm progress bars from child processes.
        if use_saved_disk_networks_checkpoints:
            from Checkpointing import (
                dqn_q_checkpoint_path,
                has_strict_field_candidate,
                pg_actor_checkpoint_path,
                pg_critic_checkpoint_path,
            )

            # Strict-field gate (training truncation) is enforced under any
            # circumstances, so the orchestrator-side report must check the
            # same gate. "Loading existing" is only announced when at least
            # one sidecar candidate matches this field against the current
            # run's value. Both PG's 'max_train_episode_length' and DQN's
            # 'max_episode_length' keys are populated to the same value for
            # cross-algo compatibility.
            strict_target = {
                "max_train_episode_length": max_train_episode_length,
                "max_episode_length": max_train_episode_length,
            }

            def _any_candidate_exists(base_path: str) -> bool:
                return has_strict_field_candidate(
                    checkpoint_path=base_path,
                    target_metadata=strict_target,
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
                kw = job["kwargs"]
                if method == "dqn":
                    dqn_path = dqn_q_checkpoint_path(
                        nn_hidden_layer_widths=kw["nn_hidden_layer_widths"],
                    ).file_path
                    if dqn_path not in reported_paths:
                        reported_paths.add(dqn_path)
                        if _any_candidate_exists(dqn_path):
                            print(f"[DQN] Loading existing Q-network checkpoint: {dqn_path}")
                        else:
                            print(f"[DQN] No checkpoint found for DQN to load from disk")
                        any_reported = True
                    continue
                if method not in pg_method_to_algo:
                    continue
                algo_type, has_critic = pg_method_to_algo[method]
                actor_path = pg_actor_checkpoint_path(
                    algo_type=algo_type,
                    actor_hidden_nn=kw["actor_hidden_nn"],
                ).file_path
                if actor_path not in reported_paths:
                    reported_paths.add(actor_path)
                    if _any_candidate_exists(actor_path):
                        print(f"[{algo_type}] Loading existing actor checkpoint: {actor_path}")
                    else:
                        print(f"[{algo_type}] No actor checkpoint found for {algo_type} to load from disk")
                    any_reported = True
                if has_critic:
                    critic_path = pg_critic_checkpoint_path(
                        algo_type=algo_type,
                        critic_hidden_nn=kw["critic_hidden_nn"],
                    ).file_path
                    if critic_path not in reported_paths:
                        reported_paths.add(critic_path)
                        if _any_candidate_exists(critic_path):
                            print(f"[{algo_type}] Loading existing critic checkpoint: {critic_path}")
                        else:
                            print(f"[{algo_type}] No critic checkpoint found for {algo_type} to load from disk")
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
            use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
            skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
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

    # ── Final summary: mean & std of returns per (algorithm, setting) ──
    try:
        build_returns_summary_table(
            algo_jobs=algo_jobs,
            setting_results=setting_results,
            algo_job_offsets=algo_job_offsets,
            n_repetitions=n_repetitions,
            last_fraction=0.1,
            output_dir="Trial Continuation Analysis",
            use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
            global_config=gc,
        )
    except Exception as exc:
        print(f"[summary] Failed to build returns summary table: {exc}")

    total_time = (time.perf_counter() - start_time) / 60.0
    with open("output.log", "w", encoding="utf-8") as f:
        f.write(f"Total execution time: {total_time:.3f} minutes\n")
    print(f"\nExperiment finished in {total_time:.3f} minutes.")

    if show_individual_plots or animation_plot or combined_fig_shown:
        plt.show(block=show_curve_plots)




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


def _checkpoint_metadata_from_kwargs(**kwargs):
    metadata = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()
        elif torch.is_tensor(value):
            tensor = value.detach().cpu()
            metadata[key] = tensor.item() if tensor.ndim == 0 else tensor.tolist()
        elif isinstance(value, (np.bool_, bool)):
            metadata[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            metadata[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            metadata[key] = float(value)
        else:
            metadata[key] = value
    return metadata


def _pg_actor_checkpoint_metadata(*, algo_type, actor_hidden_nn, actor_lr, gamma, max_train_episode_length,
                                  max_eval_episode_length, eval_with_env_episode_trials, n_eval_episodes,
                                  n_timesteps, eval_interval):
    return _checkpoint_metadata_from_kwargs(
        algo_type=algo_type,
        component="Actor",
        actor_hidden_nn=actor_hidden_nn,
        actor_lr=actor_lr,
        gamma=gamma,
        max_train_episode_length=max_train_episode_length,
        max_eval_episode_length=max_eval_episode_length,
        eval_with_env_episode_trials=eval_with_env_episode_trials,
        n_eval_episodes=n_eval_episodes,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
    )


def _pg_critic_checkpoint_metadata(*, algo_type, actor_hidden_nn, critic_hidden_nn, actor_lr, critic_lr, gamma,
                                   TN_step=None, max_train_episode_length=None, max_eval_episode_length=None,
                                   eval_with_env_episode_trials=None, n_eval_episodes=None, n_timesteps=None,
                                   eval_interval=None, gae_lambda=None, clip_eps=None, n_epochs=None,
                                   rollout_steps=None):
    return _checkpoint_metadata_from_kwargs(
        algo_type=algo_type,
        component="Critic",
        actor_hidden_nn=actor_hidden_nn,
        critic_hidden_nn=critic_hidden_nn,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        TN_step=TN_step,
        max_train_episode_length=max_train_episode_length,
        max_eval_episode_length=max_eval_episode_length,
        eval_with_env_episode_trials=eval_with_env_episode_trials,
        n_eval_episodes=n_eval_episodes,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        rollout_steps=rollout_steps,
    )


def _dqn_checkpoint_metadata(**kwargs):
    return _checkpoint_metadata_from_kwargs(**kwargs)


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
    use_saved_disk_networks_checkpoints: bool = False,
    skip_selection_hyperparameter_match: bool = False,
    eval_with_env_episode_trials: bool = False,
    n_eval_episodes: int = 5,
    # PPO-specific
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 10,
    rollout_steps: int = 2048,
    # ── Engineering-trick hyperparameters (optimal defaults; overridable from
    #    the Experiment*.py configs via the job builders + forwarding whitelist) ──
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    max_grad_norm=0.5,
    adam_eps: float = 1e-5,
    anneal_lr: bool = True,
    orthogonal_init: bool = True,
    normalize_advantages: bool = True,
    normalize_obs: bool = False,
    activation_name: str = "tanh",
    use_advantage: bool = True,
    use_gae: bool = True,
    num_minibatches: int = 32,
    clip_vloss: bool = True,
    target_kl=None,
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
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            adam_eps=adam_eps,
            anneal_lr=anneal_lr,
            orthogonal_init=orthogonal_init,
            normalize_advantages=normalize_advantages,
            normalize_obs=normalize_obs,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="REINFORCE",
            actor_hidden_nn=actor_hidden_nn,
        )
        actor_metadata = {
            "algo_type": "REINFORCE",
            "component": "Actor",
            "actor_hidden_nn": actor_hidden_nn,
            "actor_lr": actor_lr,
            "gamma": gamma,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }
        if use_saved_disk_networks_checkpoints:
            load_state_dict_if_present(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
                metadata=actor_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
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
                metadata=actor_metadata,
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
            use_advantage=use_advantage,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            adam_eps=adam_eps,
            anneal_lr=anneal_lr,
            orthogonal_init=orthogonal_init,
            normalize_advantages=normalize_advantages,
            normalize_obs=normalize_obs,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="AC",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="AC",
            critic_hidden_nn=critic_hidden_nn,
        )
        actor_metadata = {
            "algo_type": "AC",
            "component": "Actor",
            "actor_hidden_nn": actor_hidden_nn,
            "actor_lr": actor_lr,
            "gamma": gamma,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }
        critic_metadata = {
            "algo_type": "AC",
            "component": "Critic",
            "actor_hidden_nn": actor_hidden_nn,
            "critic_hidden_nn": critic_hidden_nn,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }

        if use_saved_disk_networks_checkpoints:
            load_state_dict_if_present(
                model=agent.actor,
                checkpoint_path=actor_ck.file_path,
                metadata=actor_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
            )
            load_state_dict_if_present(
                model=agent.critic,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
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
                metadata=actor_metadata,
            )
            save_state_dict_overwrite(
                model=agent.critic,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
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
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            adam_eps=adam_eps,
            anneal_lr=anneal_lr,
            orthogonal_init=orthogonal_init,
            normalize_advantages=normalize_advantages,
            normalize_obs=normalize_obs,
            activation_name=activation_name,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="A2C",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="A2C",
            critic_hidden_nn=critic_hidden_nn,
        )
        actor_metadata = {
            "algo_type": "A2C",
            "component": "Actor",
            "actor_hidden_nn": actor_hidden_nn,
            "actor_lr": actor_lr,
            "gamma": gamma,
            "TN_step": TN_step,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }
        critic_metadata = {
            "algo_type": "A2C",
            "component": "Critic",
            "actor_hidden_nn": actor_hidden_nn,
            "critic_hidden_nn": critic_hidden_nn,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "TN_step": TN_step,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }

        if use_saved_disk_networks_checkpoints:
            load_state_dict_if_present(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
                metadata=actor_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
            )
            load_state_dict_if_present(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
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
                metadata=actor_metadata,
            )
            save_state_dict_overwrite(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
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
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            adam_eps=adam_eps,
            anneal_lr=anneal_lr,
            orthogonal_init=orthogonal_init,
            normalize_advantages=normalize_advantages,
            normalize_obs=normalize_obs,
            activation_name=activation_name,
            num_minibatches=num_minibatches,
            clip_vloss=clip_vloss,
            target_kl=target_kl,
        )

        actor_ck = pg_actor_checkpoint_path(
            algo_type="PPO",
            actor_hidden_nn=actor_hidden_nn,
        )
        critic_ck = pg_critic_checkpoint_path(
            algo_type="PPO",
            critic_hidden_nn=critic_hidden_nn,
        )
        actor_metadata = {
            "algo_type": "PPO",
            "component": "Actor",
            "actor_hidden_nn": actor_hidden_nn,
            "actor_lr": actor_lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "n_epochs": n_epochs,
            "rollout_steps": rollout_steps,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }
        critic_metadata = {
            "algo_type": "PPO",
            "component": "Critic",
            "actor_hidden_nn": actor_hidden_nn,
            "critic_hidden_nn": critic_hidden_nn,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "n_epochs": n_epochs,
            "rollout_steps": rollout_steps,
            "max_train_episode_length": max_train_episode_length,
            "max_eval_episode_length": max_eval_episode_length,
            "eval_with_env_episode_trials": eval_with_env_episode_trials,
            "n_eval_episodes": n_eval_episodes,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
        }

        if use_saved_disk_networks_checkpoints:
            load_state_dict_if_present(
                model=agent.policy,
                checkpoint_path=actor_ck.file_path,
                metadata=actor_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
            )
            load_state_dict_if_present(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
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
                metadata=actor_metadata,
            )
            save_state_dict_overwrite(
                model=agent.value_func,
                checkpoint_path=critic_ck.file_path,
                metadata=critic_metadata,
            )

    return rep_returns, rep_timesteps
