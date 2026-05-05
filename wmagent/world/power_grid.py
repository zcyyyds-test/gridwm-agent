"""Power-grid adapter for the gridwm-agent world-model system API."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wmagent.data.dataset import WMAgentDataset
from wmagent.data.normalizer import invert_norm, load_stats
from wmagent.eval.risk import normalize_risk_scores, rollout_risk_features, rollout_risk_value
from wmagent.eval.scenario_search import ScenarioGrid, generate_candidate_action_sequences
from wmagent.models.factory import build_world_model
from wmagent.train.loop import load_static_graph
from wmagent.world.base import ImaginedFuture, RiskSignal, SearchResult, WorldEvent, WorldState


def _rollout_model(model, state_t: torch.Tensor, action_sequence: torch.Tensor, graph):
    if hasattr(model, "rollout"):
        return model.rollout(state_t, action_sequence=action_sequence, graph=graph)
    states = [state_t]
    s = state_t
    for step in range(action_sequence.shape[1]):
        s = model(s, action_global=action_sequence[:, step], graph=graph)
        states.append(s)
    return torch.stack(states, dim=1)


def _risk_signal_from_feature(feature: dict[str, Any]) -> RiskSignal:
    return RiskSignal(
        score=int(feature["score"]),
        band=str(feature["band"]),
        value=float(feature["risk_value"]),
        features={
            k: v
            for k, v in feature.items()
            if k not in {"score", "band", "risk_value"}
        },
    )


def _ensure_domain(obj: WorldState | WorldEvent, *, expected: str) -> None:
    if obj.domain != expected:
        raise ValueError(f"expected domain={expected!r}, got {obj.domain!r}")


class PowerGridWorldModelSystem:
    """Application-facing wrapper around a trained gridwm-agent checkpoint."""

    domain = "power_grid"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        graph,
        stats: dict[str, Any],
        device: torch.device,
        run_id: str,
        model_type: str,
        checkpoint_epoch: int | None,
        anchor_cache_path: Path | None = None,
    ) -> None:
        self.model = model
        self.graph = graph
        self.stats = stats
        self.device = device
        self.run_id = run_id
        self.model_type = model_type
        self.checkpoint_epoch = checkpoint_epoch
        self.anchor_cache_path = anchor_cache_path

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        *,
        model_cfg_fallback: Path = Path("configs/model.yaml"),
        graph_path: Path = Path("data/graph_ieee39.h5"),
        norm_stats_path: Path = Path("data/norm_stats.json"),
        device: torch.device | None = None,
    ) -> "PowerGridWorldModelSystem":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = load_static_graph(graph_path, device)
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        fallback_cfg = yaml.safe_load(model_cfg_fallback.read_text())
        model_cfg = dict(ckpt.get("cfg") or fallback_cfg)
        model_cfg.setdefault("model_type", ckpt.get("model_type", "mpnn"))
        model = build_world_model(model_cfg, graph).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        anchor_cache_path = run_dir / "anchors.pt"
        return cls(
            model=model,
            graph=graph,
            stats=load_stats(norm_stats_path),
            device=device,
            run_id=run_dir.name,
            model_type=str(model_cfg.get("model_type", ckpt.get("model_type", "unknown"))),
            checkpoint_epoch=ckpt.get("epoch"),
            anchor_cache_path=anchor_cache_path if anchor_cache_path.exists() else None,
        )

    def anchor_state(
        self,
        *,
        split: str = "val",
        seed: int = 2026,
        horizon: int = 10,
        anchor_index: int = 0,
    ) -> WorldState:
        # Cache fast-path: if the run_dir ships an anchor cache and the
        # request matches a cached entry, use it. This lets the bundled
        # demo_run quickstart work on a fresh clone with no `data/raw/`.
        cached = self._lookup_cached_anchor(
            split=split, seed=seed, horizon=horizon, anchor_index=anchor_index,
        )
        if cached is not None:
            return cached

        # Fallback: build from raw HDF5 trajectories. Requires the user to
        # have run `scripts/collect_data.py` (see README -> Data).
        if not Path("data/raw").exists() or not any(Path("data/raw").glob("*.h5")):
            raise FileNotFoundError(
                "anchor_state needs either a cached anchor in "
                "f'{run_dir}/anchors.pt' (matching split/seed/horizon/"
                "anchor_index) or raw trajectories in data/raw/. "
                f"Cache miss for (split={split!r}, seed={seed}, "
                f"horizon={horizon}, anchor_index={anchor_index}). "
                "Re-collect data or rebuild the cache via "
                "scripts/build_demo_anchors.py."
            )
        train_cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
        dataset = WMAgentDataset(
            raw_dir=Path("data/raw"),
            splits_path=Path("data/splits.json"),
            norm_stats_path=Path("data/norm_stats.json"),
            split=split,  # type: ignore[arg-type]
            pairs_per_traj_per_epoch=max(
                anchor_index + 1,
                int(train_cfg["pairs_per_traj_per_epoch"]),
            ),
            seed=seed,
            fault_window_frac=1.0,
            rollout_horizon=horizon,
        )
        item = dataset[anchor_index]
        return WorldState(
            tensor=item["state_t"],
            domain=self.domain,
            metadata={
                "split": split,
                "seed": seed,
                "horizon": horizon,
                "anchor_index": anchor_index,
                "action_global": item["action_global"].cpu().tolist(),
                "fault_window_active": float(item["fault_window_active"]),
            },
        )

    def _lookup_cached_anchor(
        self, *, split: str, seed: int, horizon: int, anchor_index: int,
    ) -> WorldState | None:
        if self.anchor_cache_path is None or not self.anchor_cache_path.exists():
            return None
        payload = torch.load(self.anchor_cache_path, map_location="cpu", weights_only=True)
        for entry in payload.get("anchors", []):
            if (
                entry["split"] == split
                and entry["seed"] == seed
                and entry["horizon"] == horizon
                and entry["anchor_index"] == anchor_index
            ):
                return WorldState(
                    tensor=entry["state_t"],
                    domain=self.domain,
                    metadata={
                        "split": split,
                        "seed": seed,
                        "horizon": horizon,
                        "anchor_index": anchor_index,
                        "action_global": entry["action_global"].cpu().tolist(),
                        "fault_window_active": float(entry["fault_window_active"]),
                        "source": "demo_anchor_cache",
                    },
                )
        return None

    def candidate_events(
        self,
        *,
        horizon: int,
        grid: ScenarioGrid | None = None,
    ) -> list[WorldEvent]:
        action_sequences, metadata = generate_candidate_action_sequences(
            horizon=horizon,
            grid=grid or ScenarioGrid(),
            device=self.device,
        )
        return [
            WorldEvent(tensor=action_sequences[i], domain=self.domain, metadata=metadata[i])
            for i in range(action_sequences.shape[0])
        ]

    def imagine(self, state: WorldState, event: WorldEvent) -> ImaginedFuture:
        return self.imagine_many(state, [event])[0]

    def imagine_many(
        self,
        state: WorldState,
        events: list[WorldEvent],
        *,
        batch_size: int = 64,
    ) -> list[ImaginedFuture]:
        if not events:
            return []
        _ensure_domain(state, expected=self.domain)
        for event in events:
            _ensure_domain(event, expected=self.domain)
        state_tensor = state.tensor.to(self.device)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        action_sequences = torch.stack([event.tensor.to(self.device) for event in events], dim=0)
        chunks = []
        with torch.no_grad():
            for start in range(0, action_sequences.shape[0], batch_size):
                actions = action_sequences[start: start + batch_size]
                states = state_tensor.expand(actions.shape[0], -1, -1).contiguous()
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    pred_norm = _rollout_model(self.model, states, actions, self.graph)
                pred_np = invert_norm(pred_norm.float().cpu().numpy(), self.stats)
                pred_phys = torch.from_numpy(pred_np)
                chunks.append(pred_phys)
        rollouts = torch.cat(chunks, dim=0)
        return [
            ImaginedFuture(rollout=rollouts[i], state=state, event=events[i])
            for i in range(rollouts.shape[0])
        ]

    def score(self, future: ImaginedFuture) -> RiskSignal:
        feature = rollout_risk_features(future.rollout.unsqueeze(0))[0]
        return _risk_signal_from_feature(feature)

    def search(
        self,
        state: WorldState,
        events: list[WorldEvent],
        *,
        top_k: int = 10,
        batch_size: int = 64,
    ) -> list[SearchResult]:
        _ensure_domain(state, expected=self.domain)
        for event in events:
            _ensure_domain(event, expected=self.domain)
        futures = self.imagine_many(state, events, batch_size=batch_size)
        if not futures:
            return []
        rollouts = torch.stack([future.rollout for future in futures], dim=0)
        raw = rollout_risk_value(rollouts)
        scores = normalize_risk_scores(raw)
        features = rollout_risk_features(rollouts, scores=scores)
        order = torch.argsort(raw, descending=True).tolist()
        results = []
        for rank, idx in enumerate(order[:top_k], start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    future=futures[idx],
                    risk=_risk_signal_from_feature(features[idx]),
                )
            )
        return results


def search_result_to_record(
    result: SearchResult,
    *,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    row = {
        "rank": result.rank,
        "scenario": result.future.event.metadata,
        "risk": {
            "score": result.risk.score,
            "band": result.risk.band,
            "risk_value": result.risk.value,
            **result.risk.features,
        },
    }
    if candidate_index is not None:
        row["candidate_index"] = candidate_index
    return row


def aggregate_risk_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_fault[row["scenario"]["event_code"]].append(row)
        by_channel[row["risk"]["dominant_channel"]].append(row)
        by_node[str(row["risk"]["dominant_node"])].append(row)

    def summarize(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        rows = []
        for key, vals in grouped.items():
            raw = np.array([v["risk"]["risk_value"] for v in vals], dtype="float64")
            scores = np.array([v["risk"]["score"] for v in vals], dtype="float64")
            rows.append(
                {
                    "key": key,
                    "count": int(len(vals)),
                    "mean_score": round(float(scores.mean()), 2),
                    "max_score": int(scores.max()),
                    "mean_risk_value": round(float(raw.mean()), 6),
                    "max_risk_value": round(float(raw.max()), 6),
                }
            )
        rows.sort(key=lambda r: (r["max_score"], r["mean_score"], r["count"]), reverse=True)
        return rows

    return {
        "by_fault": summarize(by_fault),
        "by_dominant_channel": summarize(by_channel),
        "by_dominant_node": summarize(by_node),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
