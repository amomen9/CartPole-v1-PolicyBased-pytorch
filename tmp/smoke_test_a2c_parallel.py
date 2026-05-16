import os
import sys
import numpy as np

# Ensure repo root is on sys.path when running `python tmp/<script>.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import Library  # noqa: E402


if __name__ == "__main__":
    results = Library.average_over_repetitions(
        method="a2c",
        n_repetitions=2,
        n_timesteps=500,
        eval_interval=100,
        max_episode_length=50,  # used as truncation_step in run_a2c
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        actor_hidden_nn=np.array([8]),
        critic_hidden_nn=np.array([8]),
        base_seed=123,
        plot_smoothing_window=None,
        eval_with_env_episode_trials=False,
        n_eval_episodes=1,
        return_raw=False,
    )
    mean, std, timesteps = results
    print("SMOKE_OK", len(mean), len(std), len(timesteps), float(mean[-1]))
