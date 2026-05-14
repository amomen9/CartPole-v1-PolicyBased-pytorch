import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointPaths:
    dir_path: str
    file_path: str

    def ensure_dir(self) -> None:
        os.makedirs(self.dir_path, exist_ok=True)


def _as_int_vector(x: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("Hidden-layer widths must be non-empty.")
    # Flatten and cast to int
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    return arr


def architecture_signature(hidden_layer_widths: Sequence[int] | np.ndarray) -> str:
    """
    Stable string signature based on hidden-layer widths only.
    Example: [64, 64] -> '64-64'
    """
    arr = _as_int_vector(hidden_layer_widths)
    return "-".join(str(int(v)) for v in arr)


def _algo_dir(algo_type: str) -> str:
    # Uses the already pre-created sub-directories in the repo.
    return os.path.join("Checkpoints", algo_type)


def _pg_component_dir(algo_type: str, component: str) -> str:
    return os.path.join(_algo_dir(algo_type), component)


def pg_actor_checkpoint_path(*, algo_type: str, actor_hidden_nn: Sequence[int] | np.ndarray) -> CheckpointPaths:
    sig = architecture_signature(actor_hidden_nn)
    dir_path = _pg_component_dir(algo_type, "Actor")
    file_path = os.path.join(dir_path, f"actor_{sig}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


def pg_critic_checkpoint_path(*, algo_type: str, critic_hidden_nn: Sequence[int] | np.ndarray) -> CheckpointPaths:
    sig = architecture_signature(critic_hidden_nn)
    dir_path = _pg_component_dir(algo_type, "Critic")
    file_path = os.path.join(dir_path, f"critic_{sig}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


def load_state_dict_if_present(*, model: torch.nn.Module, checkpoint_path: str) -> bool:
    if not os.path.isfile(checkpoint_path):
        return False
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return True


def save_state_dict_overwrite(*, model: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Atomically overwrite checkpoint on disk.

    Important for parallel training where multiple reps might finish close
    together. We write to a temp file and then os.replace() it.
    """
    dir_name = os.path.dirname(checkpoint_path)
    os.makedirs(dir_name, exist_ok=True)

    tmp_path = checkpoint_path + f".tmp_{os.getpid()}"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, checkpoint_path)


def dqn_q_checkpoint_path(*, nn_hidden_layer_widths: Sequence[int] | np.ndarray) -> CheckpointPaths:
    sig = architecture_signature(nn_hidden_layer_widths)
    dir_path = _algo_dir("DQN")
    file_path = os.path.join(dir_path, f"dqn_q_{sig}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


def sac_q_checkpoint_path(
    *,
    critic_hidden_nn: Sequence[int] | np.ndarray,
    q_index: int,
) -> CheckpointPaths:
    sig = architecture_signature(critic_hidden_nn)
    component = f"Q{int(q_index)}"
    dir_path = _pg_component_dir("SAC", component)
    file_path = os.path.join(dir_path, f"q{int(q_index)}_{sig}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)
