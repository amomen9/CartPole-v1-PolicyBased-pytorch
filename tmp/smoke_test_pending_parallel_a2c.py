import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from facilitation_functions import _run_pending_parallel


if __name__ == "__main__":
    # Minimal pending_settings entry that triggers the _run_single_repetition()
    # A2C code path inside facilitation_functions._run_pending_parallel.
    pending_settings = [
        (
            0,
            {
                "curve_label": "SMOKE_A2C",
                "method": "a2c",
                "kwargs": {
                    "actor_hidden_nn": np.array([8], dtype=np.int32),
                    "critic_hidden_nn": np.array([8], dtype=np.int32),
                    "actor_lr": 1e-3,
                    "critic_lr": 1e-3,
                    "gamma": 0.99,
                    "eval_with_env_episode_trials": False,
                    "n_eval_episodes": 1,
                },
            },
        )
    ]

    setting_results = [None] * len(pending_settings)

    _run_pending_parallel(
        pending_settings=pending_settings,
        n_repetitions=2,
        n_timesteps=300,
        eval_interval=100,
        max_train_episode_length=50,
        max_eval_episode_length=50,
        base_seed=123,
        use_existing_network_checkpoints=False,
        setting_results=setting_results,
    )

    lc_mean, lc_std, timesteps = setting_results[0]
    print("PENDING_PARALLEL_OK", len(lc_mean), len(lc_std), len(timesteps), float(lc_mean[-1]))
