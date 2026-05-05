"""HDF5 sample schema for wm-agent Phase 1 trajectories.

Per the spec §3.5: each trajectory is one h5py group `sample_<uid>/` with
fixed sub-structure. Channel order on `state` is [wr, LA, VT, IT].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

CHANNELS = ("wr", "LA", "VT", "IT")
N_GENERATORS = 10
N_BUS_OBS = 3


@dataclass(frozen=True)
class FaultAction:
    fs: float           # fault start, seconds, in [1.0, 5.0]
    fe: float           # fault end, seconds; fe - fs in [0.05, 0.20]
    ft: int             # fault type code; Phase 1 ∈ {1, 3, 7}
    chg_ohm: float      # fault resistance, Ω, in [0.01, 10]

    def __post_init__(self) -> None:
        if not (1.0 <= self.fs <= 5.0):
            raise ValueError(f"fs must be in [1.0, 5.0]; got {self.fs}")
        d = self.fe - self.fs
        if not (0.05 <= d <= 0.20):
            raise ValueError(f"duration fe-fs must be in [0.05, 0.20]; got {d}")
        if self.ft not in {1, 3, 7}:
            raise ValueError(f"ft must be in {{1, 3, 7}} in Phase 1; got {self.ft}")
        if not (0.01 <= self.chg_ohm <= 10.0):
            raise ValueError(f"chg_ohm must be in [0.01, 10.0]; got {self.chg_ohm}")


@dataclass(frozen=True)
class SampleGroup:
    uid: str
    action: FaultAction
    state: np.ndarray  # (T, N=10, C=4) float32
    bus_obs: np.ndarray  # (T, 3) float32
    meta: dict[str, Any]


REQUIRED_META_KEYS = {
    "case", "seed", "cloudpss_rid", "cloudpss_version", "data_version",
    "output_dt_seconds", "topology_static_during_fault", "collected_at",
}


def validate_sample(sg: SampleGroup, *, expected_n_samples: int) -> None:
    if sg.state.dtype != np.float32:
        raise ValueError(f"state dtype must be float32; got {sg.state.dtype}")
    if sg.state.shape != (expected_n_samples, N_GENERATORS, len(CHANNELS)):
        raise ValueError(
            f"state shape must be ({expected_n_samples},{N_GENERATORS},"
            f"{len(CHANNELS)}); got {sg.state.shape}"
        )
    if sg.bus_obs.shape != (expected_n_samples, N_BUS_OBS):
        raise ValueError(f"bus_obs shape must be ({expected_n_samples},3); got {sg.bus_obs.shape}")
    missing = REQUIRED_META_KEYS - sg.meta.keys()
    if missing:
        raise ValueError(f"meta missing keys: {missing}")


def write_sample(f: h5py.File, sg: SampleGroup) -> None:
    g = f.create_group(f"sample_{sg.uid}")
    meta = g.create_group("meta")
    for k, v in sg.meta.items():
        meta.attrs[k] = v
    action = g.create_group("action")
    action.attrs["fs"] = sg.action.fs
    action.attrs["fe"] = sg.action.fe
    action.attrs["ft"] = sg.action.ft
    action.attrs["chg_ohm"] = sg.action.chg_ohm
    state = g.create_group("state")
    for i, ch in enumerate(CHANNELS):
        state.create_dataset(ch, data=sg.state[..., i], compression="gzip", compression_opts=4)
    g.create_dataset("bus_obs", data=sg.bus_obs, compression="gzip", compression_opts=4)


def read_sample(f: h5py.File, uid: str) -> SampleGroup:
    g = f[f"sample_{uid}"]
    meta = dict(g["meta"].attrs)
    a = g["action"].attrs
    action = FaultAction(
        fs=float(a["fs"]),
        fe=float(a["fe"]),
        ft=int(a["ft"]),
        chg_ohm=float(a["chg_ohm"]),
    )
    state_g = g["state"]
    n_samples = state_g[CHANNELS[0]].shape[0]
    state = np.empty((n_samples, N_GENERATORS, len(CHANNELS)), dtype="float32")
    for i, ch in enumerate(CHANNELS):
        state[..., i] = state_g[ch][...]
    bus_obs = g["bus_obs"][...]
    return SampleGroup(uid=uid, action=action, state=state, bus_obs=bus_obs, meta=meta)
