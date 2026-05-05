"""PyTorch Dataset over collected gridwm-agent HDF5 trajectories (path-A).

Each item is a short rollout window from a randomly chosen trajectory in the
chosen split, at a random t. State is normalized; the first pair stays
available for one-step baselines, while `state_sequence`, `action_sequence`,
and `fault_window_sequence` drive latent world-model rollout training.

Stage 2 changes (D22 / DeepSeek's option D):
- Stratified sampling: with probability `fault_window_frac` (default 0.30),
  the chosen `t` is forced to fall inside the fault window, otherwise it's
  uniformly sampled across the whole trajectory. Fixes the 96/4 imbalance
  that caused V5's mode-collapse-to-zero.
- Time encoding: action_global gains a 12th dim `tau = (t - fs) / duration`
  clipped to [-2, 3] so the model can tell pre-fault / fault-active /
  post-fault apart from action + state alone.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from wmagent.data.normalizer import apply_norm, load_stats
from wmagent.data.schema import CHANNELS, FaultAction, read_sample
from wmagent.data.splits import load_split


SplitName = Literal["train", "val", "test"]

# Action vector layout: 8 ft one-hot + fs_norm + duration_norm + log10(chg)_norm
# + tau (current time relative to fault, clipped to [-2,3]) = 12.
ACTION_DIM = 12
_FS_MIN, _FS_MAX = 1.0, 5.0
_DUR_MIN, _DUR_MAX = 0.05, 0.20
_CHG_LOG_MIN = math.log10(0.01)
_CHG_LOG_MAX = math.log10(10.0)
_TAU_CLIP = (-2.0, 3.0)


def encode_action_global(action: FaultAction, *, tau: float) -> np.ndarray:
    """Build the 12-d global action vector. `tau = (t - fs) / duration` —
    < 0 pre-fault, in [0,1] during fault, > 1 post-fault.
    """
    out = np.zeros(ACTION_DIM, dtype="float32")
    out[action.ft] = 1.0  # ft one-hot at indices 0..7; Phase 1 uses {1,3,7}
    out[8] = (action.fs - _FS_MIN) / (_FS_MAX - _FS_MIN)
    duration = action.fe - action.fs
    out[9] = (duration - _DUR_MIN) / (_DUR_MAX - _DUR_MIN)
    log_chg = math.log10(max(action.chg_ohm, 1e-12))
    out[10] = (log_chg - _CHG_LOG_MIN) / (_CHG_LOG_MAX - _CHG_LOG_MIN)
    out[11] = float(np.clip(tau, _TAU_CLIP[0], _TAU_CLIP[1]))
    return out


class WMAgentDataset(Dataset):
    def __init__(
        self,
        *,
        raw_dir: Path,
        splits_path: Path,
        norm_stats_path: Path,
        split: SplitName,
        pairs_per_traj_per_epoch: int = 64,
        seed: int = 0,
        fault_window_frac: float = 0.30,
        post_clear_window_s: float = 0.5,
        rollout_horizon: int = 1,
    ) -> None:
        """`fault_window_frac` is the per-item probability of forcing `t` into
        the fault window [fs, fe + post_clear_window_s]. With ~4% of timesteps
        naturally in the fault window across the dataset, setting this to 0.30
        gives ~7.5× upsampling of fault dynamics — enough to keep the gradient
        signal alive without entirely abandoning steady-state coverage.
        """
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.split_def = load_split(Path(splits_path))
        self.stats = load_stats(Path(norm_stats_path))
        self.pairs_per_traj = pairs_per_traj_per_epoch
        self.rng = np.random.default_rng(seed)
        self.fault_window_frac = float(fault_window_frac)
        self.post_clear_window_s = float(post_clear_window_s)
        self.rollout_horizon = max(1, int(rollout_horizon))

        self.entries: list[tuple[Path, str]] = []
        for h5_path in sorted(self.raw_dir.glob("*.h5")):
            with h5py.File(h5_path, "r") as f:
                for uid_full in f.keys():
                    uid = uid_full.removeprefix("sample_")
                    if self.split_def.split_for_uid(uid) == self.split:
                        self.entries.append((h5_path, uid))

    def __len__(self) -> int:
        return len(self.entries) * self.pairs_per_traj

    def _sample_t(self, T: int, sg, output_dt_s: float) -> tuple[int, bool]:
        """Sample a timestep. With prob `fault_window_frac`, force into the
        fault window; otherwise uniform over the whole trajectory.
        Returns (t_index, is_fault_window).
        """
        max_start = T - self.rollout_horizon - 1
        if max_start < 1:
            raise ValueError(
                f"trajectory too short for rollout_horizon={self.rollout_horizon}: T={T}"
            )
        force_fault = (self.rng.random() < self.fault_window_frac)
        if force_fault:
            fs, fe = sg.action.fs, sg.action.fe
            t_lo = max(1, int(np.floor(fs / output_dt_s)))
            t_hi = min(
                max_start,
                int(np.ceil((fe + self.post_clear_window_s) / output_dt_s)),
            )
            if t_hi <= t_lo:
                # Fallback: window collapsed (shouldn't happen with realistic params)
                return int(self.rng.integers(1, max_start + 1)), False
            t = int(self.rng.integers(t_lo, t_hi + 1))
            return t, True
        return int(self.rng.integers(1, max_start + 1)), False

    def __getitem__(self, idx: int) -> dict:
        h5_path, uid = self.entries[idx % len(self.entries)]
        with h5py.File(h5_path, "r") as f:
            sg = read_sample(f, uid)
        T = sg.state.shape[0]
        output_dt_s = float(sg.meta.get("output_dt_seconds", 0.001))
        t, _force_was_fault = self._sample_t(T, sg, output_dt_s)

        state_tm1 = apply_norm(sg.state[t - 1], self.stats)
        raw_seq = sg.state[t: t + self.rollout_horizon + 1]
        state_seq = apply_norm(raw_seq, self.stats).astype("float32", copy=False)
        s_t = state_seq[0]
        s_tp1 = state_seq[1]

        # Δ in state-norm units so `s_t + Δ = s_tp1` residual is unit-consistent
        # with WorldModel.forward.
        delta_norm = (s_tp1 - s_t).astype("float32", copy=False)
        delta_sequence = (state_seq[1:] - state_seq[:-1]).astype("float32", copy=False)

        # Time-relative-to-fault scalar for action_global[11].
        fs, fe = sg.action.fs, sg.action.fe
        duration = max(fe - fs, 1e-6)
        action_sequence = []
        fault_window_sequence = []
        for step in range(self.rollout_horizon):
            t_step_phys = (t + step) * output_dt_s
            tau = (t_step_phys - fs) / duration
            action_sequence.append(encode_action_global(sg.action, tau=tau))
            in_fault_window = fs <= t_step_phys <= (fe + self.post_clear_window_s)
            fault_window_sequence.append(float(in_fault_window))
        action_sequence_arr = np.stack(action_sequence, axis=0).astype("float32", copy=False)
        action_global = action_sequence_arr[0]

        # fault_window_active flag is True iff t is inside the fault window
        # [fs, fe + 0.5s]. Used by eval.py to compute the WM-meaningful MAE
        # (the H1 gate is on this subset, not the full set).
        fault_window_active = fault_window_sequence[0]

        return {
            "state_t": torch.from_numpy(s_t),
            "state_tm1": torch.from_numpy(state_tm1),
            "state_tp1": torch.from_numpy(s_tp1),
            "delta_norm": torch.from_numpy(delta_norm),
            "delta_sequence": torch.from_numpy(delta_sequence),
            "action_global": torch.from_numpy(action_global),
            "state_sequence": torch.from_numpy(state_seq),
            "action_sequence": torch.from_numpy(action_sequence_arr),
            "fault_window_sequence": torch.tensor(fault_window_sequence, dtype=torch.float32),
            "fault_window_active": fault_window_active,
        }
