import json
import os
from dataclasses import dataclass
from typing import Sequence

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


def dqn_q_checkpoint_path(*, nn_hidden_layer_widths: Sequence[int] | np.ndarray) -> CheckpointPaths:
    sig = architecture_signature(nn_hidden_layer_widths)
    dir_path = _algo_dir("DQN")
    file_path = os.path.join(dir_path, f"dqn_q_{sig}.pt")
    return CheckpointPaths(dir_path=dir_path, file_path=file_path)


def _metadata_path_for_checkpoint(checkpoint_path: str) -> str:
    root, _ext = os.path.splitext(checkpoint_path)
    return f"{root}.txt"


def _normalize_metadata_value(value):
    if isinstance(value, np.ndarray):
        return [_normalize_metadata_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_metadata_value(item) for item in value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _normalize_metadata_value(value.item())
        return _normalize_metadata_value(value.detach().cpu().tolist())
    return value


def _normalize_metadata(metadata):
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict or None")
    return {str(key): _normalize_metadata_value(value) for key, value in metadata.items()}


def _metadata_to_text(metadata) -> str:
    normalized = _normalize_metadata(metadata)
    if normalized is None:
        return ""
    return json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _read_metadata_file(metadata_path: str):
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    if not raw_text:
        return {}
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Checkpoint metadata file '{metadata_path}' does not contain a JSON object.")
    return _normalize_metadata(parsed)


def _write_metadata_file(metadata_path: str, metadata) -> None:
    tmp_path = metadata_path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(_metadata_to_text(metadata))
    os.replace(tmp_path, metadata_path)


def _checkpoint_stem(checkpoint_path: str) -> str:
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def _checkpoint_index_for_candidate(candidate_path: str, base_stem: str) -> int:
    candidate_stem = _checkpoint_stem(candidate_path)
    if candidate_stem == base_stem:
        return 0
    prefix = f"{base_stem}_"
    if candidate_stem.startswith(prefix):
        suffix = candidate_stem[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    return -1


def _iter_candidate_checkpoint_paths(checkpoint_path: str):
    base_dir = os.path.dirname(checkpoint_path)
    base_stem = _checkpoint_stem(checkpoint_path)
    seen: set[str] = set()

    if os.path.isfile(checkpoint_path):
        seen.add(os.path.abspath(checkpoint_path))
        yield checkpoint_path

    if not os.path.isdir(base_dir):
        return

    prefix = f"{base_stem}_"
    for name in sorted(os.listdir(base_dir)):
        if not name.endswith(".pt"):
            continue
        candidate_path = os.path.join(base_dir, name)
        abs_candidate = os.path.abspath(candidate_path)
        if abs_candidate in seen:
            continue
        candidate_stem = _checkpoint_stem(candidate_path)
        if candidate_stem == base_stem or candidate_stem.startswith(prefix):
            seen.add(abs_candidate)
            yield candidate_path


def _resolve_non_overwriting_path(checkpoint_path: str) -> str:
    if not os.path.exists(checkpoint_path):
        return checkpoint_path

    root, ext = os.path.splitext(checkpoint_path)
    index = 1
    while True:
        candidate = f"{root}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


_LOOSE_TIMESTEPS_KEYS = ("n_timesteps", "n_env_steps")
_EXACT_MATCH_EXCLUDED_KEYS = ("use_saved_disk_networks_checkpoints",)


def _strip_exact_match_excluded(metadata: dict | None) -> dict | None:
    """Drop keys that must not participate in exact-match equality (e.g. the
    'use_saved_disk_networks_checkpoints' provenance flag, which is recorded
    in the sidecar for loose-match preference but is irrelevant to weight
    compatibility)."""
    if not isinstance(metadata, dict):
        return metadata
    return {k: v for k, v in metadata.items() if k not in _EXACT_MATCH_EXCLUDED_KEYS}


def _extract_loose_timesteps(candidate_metadata: dict | None) -> int | None:
    """Return the first present timesteps-like field from a candidate sidecar."""
    if not isinstance(candidate_metadata, dict):
        return None
    for key in _LOOSE_TIMESTEPS_KEYS:
        if key in candidate_metadata:
            try:
                return int(candidate_metadata[key])
            except (TypeError, ValueError):
                continue
    return None


def _resolve_loose_checkpoint_path(checkpoint_path: str) -> str | None:
    """
    Two-tier loose lookup among candidates sharing the architecture-signature
    filename:
      1. Prefer candidates whose sidecar has
         use_saved_disk_networks_checkpoints == True (i.e. produced by a
         continuation run): pick the one with the largest n_timesteps
         (fallback n_env_steps for DQN).
      2. Fall back to all remaining candidates with numeric n_timesteps:
         pick the one with the largest value.
    Ties broken by max index then max mtime.

    Candidates without a readable sidecar or a numeric timesteps field are
    ignored. Returns None if no candidate is selectable.
    """
    base_stem = _checkpoint_stem(checkpoint_path)
    preferred: list[tuple[int, int, float, str]] = []
    fallback: list[tuple[int, int, float, str]] = []
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        metadata_path = _metadata_path_for_checkpoint(candidate_path)
        try:
            candidate_metadata = _read_metadata_file(metadata_path)
        except Exception:
            continue
        timesteps = _extract_loose_timesteps(candidate_metadata)
        if timesteps is None:
            continue
        try:
            mtime = os.path.getmtime(candidate_path)
        except OSError:
            mtime = 0.0
        entry = (timesteps, index, mtime, candidate_path)
        if (
            isinstance(candidate_metadata, dict)
            and bool(candidate_metadata.get("use_saved_disk_networks_checkpoints", False))
        ):
            preferred.append(entry)
        else:
            fallback.append(entry)

    pool = preferred if preferred else fallback
    if not pool:
        return None
    pool.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return pool[0][3]


def resolve_matching_checkpoint_path(
    *,
    checkpoint_path: str,
    metadata: dict | None = None,
    skip_selection_hyperparameter_match: bool = False,
) -> str | None:
    """
    Resolve the exact checkpoint file that matches the supplied metadata.

    If metadata is None, behaves like the legacy path-based lookup.
    If metadata is provided, the checkpoint must have a sidecar .txt file whose
    JSON content matches the supplied metadata exactly after normalization.

    If 'skip_selection_hyperparameter_match' is True, ignore the exact metadata
    match and instead pick the candidate (same architecture-signature filename)
    whose sidecar has the largest n_timesteps (fallback n_env_steps).
    """
    if skip_selection_hyperparameter_match:
        return _resolve_loose_checkpoint_path(checkpoint_path)

    if metadata is None:
        return checkpoint_path if os.path.isfile(checkpoint_path) else None

    target_metadata = _strip_exact_match_excluded(_normalize_metadata(metadata))
    if target_metadata is None:
        return checkpoint_path if os.path.isfile(checkpoint_path) else None

    candidates = []
    base_stem = _checkpoint_stem(checkpoint_path)
    for candidate_path in _iter_candidate_checkpoint_paths(checkpoint_path):
        index = _checkpoint_index_for_candidate(candidate_path, base_stem)
        if index < 0:
            continue
        metadata_path = _metadata_path_for_checkpoint(candidate_path)
        try:
            candidate_metadata = _read_metadata_file(metadata_path)
        except Exception:
            continue
        if _strip_exact_match_excluded(candidate_metadata) == target_metadata:
            try:
                mtime = os.path.getmtime(candidate_path)
            except OSError:
                mtime = 0.0
            candidates.append((index, mtime, candidate_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_state_dict_if_present(
    *,
    model: torch.nn.Module,
    checkpoint_path: str,
    metadata: dict | None = None,
    skip_selection_hyperparameter_match: bool = False,
) -> bool:
    resolved_path = resolve_matching_checkpoint_path(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
    )
    if resolved_path is None:
        return False
    state = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return True


def save_state_dict_overwrite(*, model: torch.nn.Module, checkpoint_path: str, metadata: dict | None = None) -> None:
    """
    Save a checkpoint without overwriting older checkpoints.

    If the requested checkpoint already exists, it is saved as
    '<stem>_1.pt', '<stem>_2.pt', ... and a matching '<stem>_N.txt'
    metadata sidecar is written beside it.
    """
    dir_name = os.path.dirname(checkpoint_path)
    os.makedirs(dir_name, exist_ok=True)

    output_path = _resolve_non_overwriting_path(checkpoint_path)
    tmp_path = output_path + f".tmp_{os.getpid()}"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, output_path)

    if metadata is not None:
        _write_metadata_file(_metadata_path_for_checkpoint(output_path), metadata)
