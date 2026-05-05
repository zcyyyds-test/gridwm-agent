"""Train a Dreamer-style actor/critic agent on frozen wm-agent imagination."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch

from dataclasses import replace as dataclass_replace

from wmagent.agent.data import CandidateEventSpace, build_imagination_dataset
from wmagent.agent.model import DreamerStyleAgent
from wmagent.agent.training import (
    AgentTrainConfig,
    evaluate_dataset,
    set_agent_seed,
    train_agent_epoch,
)
from wmagent.eval.scenario_search import ScenarioGrid
from wmagent.train.logging import setup_run_logger
from wmagent.world.power_grid import PowerGridWorldModelSystem


def _parse_fault_type_csv(text: str | None) -> tuple[int, ...] | None:
    if text is None or text.strip() == "":
        return None
    return tuple(int(token.strip()) for token in text.split(",") if token.strip())


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-run-dir", type=Path, default=Path("outputs/run_3acec7d14d"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--n-train-anchors", type=int, default=64)
    parser.add_argument("--n-val-anchors", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--actor-loss-weight", type=float, default=0.5)
    parser.add_argument("--critic-loss-weight", type=float, default=5.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--soft-policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--safety-loss-weight", type=float, default=1.0)
    parser.add_argument("--safety-oversample-low-band", type=float, default=0.0)
    parser.add_argument("--critic-pearson-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--init-pearson-min", type=float, default=0.05,
        help="If init critic Pearson on val < this, re-init with new sub-seed; "
             "5-seed sweep showed 2/5 collapse traces back to bad init geometry. "
             "Set to a negative value to disable.",
    )
    parser.add_argument(
        "--init-pearson-max-retries", type=int, default=8,
        help="Max re-inits before giving up and proceeding with last attempt.",
    )
    parser.add_argument(
        "--mid-train-min-pearson", type=float, default=0.05,
        help="At end of critic warmup, if val critic Pearson < this, re-init "
             "critic head + value prior in place and continue (without the "
             "wasted full-260-epoch run that the cross-seed sweep showed "
             "collapsed init geometry produces). Set to a negative value to "
             "disable.",
    )
    parser.add_argument(
        "--mid-train-max-restarts", type=int, default=2,
        help="Max critic-head re-inits during training before giving up.",
    )
    parser.add_argument(
        "--critic-warmup-epochs",
        type=int,
        default=10,
        help="During the first N epochs, train only the critic with SmoothL1 "
             "(actor_loss_weight=0, ranking_loss_weight=0). Stops the actor and "
             "ranking objective from reinforcing a randomly negative critic init "
             "(observed: 2/5 cross-seed runs collapse to Pearson < 0 with default).",
    )
    parser.add_argument(
        "--no-candidate-prior",
        action="store_true",
        help="V4.2 mode: drop the per-candidate-position priors so the agent is "
             "fully content-based and supports variable K at eval time.",
    )
    parser.add_argument(
        "--candidate-fault-types",
        type=str,
        default=None,
        help="Comma-separated fault types (e.g. '1,3') to keep in the candidate "
             "event space. Default uses the full grid (1,3,7) → 162 candidates.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_dir = args.output_root / f"agent_{uuid.uuid4().hex[:10]}"
    log = setup_run_logger(run_dir, name=f"wmagent.agent.{run_dir.name}")
    cfg = AgentTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        dropout=args.dropout,
        actor_loss_weight=args.actor_loss_weight,
        critic_loss_weight=args.critic_loss_weight,
        ranking_loss_weight=args.ranking_loss_weight,
        soft_policy_loss_weight=args.soft_policy_loss_weight,
        safety_loss_weight=args.safety_loss_weight,
        safety_oversample_low_band=args.safety_oversample_low_band,
        critic_pearson_loss_weight=args.critic_pearson_loss_weight,
        seed=args.seed,
    )
    set_agent_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} run_dir={run_dir}")
    log.info(f"config={cfg}")

    system = PowerGridWorldModelSystem.from_run_dir(args.world_run_dir, device=device)
    fault_types = _parse_fault_type_csv(args.candidate_fault_types)
    grid = dataclass_replace(ScenarioGrid(), fault_types=fault_types) if fault_types else None
    event_space = CandidateEventSpace.from_grid(horizon=args.horizon, grid=grid, device=device)
    log.info(
        f"world_run={system.run_id} candidates={event_space.n_candidates} horizon={args.horizon} "
        f"fault_types={fault_types or 'default(1,3,7)'} no_prior={args.no_candidate_prior}"
    )

    t0 = time.perf_counter()
    train_ds = build_imagination_dataset(
        system,
        split="train",
        n_anchors=args.n_train_anchors,
        seed=args.seed,
        horizon=args.horizon,
        event_space=event_space,
        rollout_batch_size=args.rollout_batch_size,
    )
    train_build_s = time.perf_counter() - t0
    log.info(f"built train imagination labels in {train_build_s:.2f}s")

    t0 = time.perf_counter()
    val_ds = build_imagination_dataset(
        system,
        split="val",
        n_anchors=args.n_val_anchors,
        seed=args.seed + 1,
        horizon=args.horizon,
        event_space=event_space,
        rollout_batch_size=args.rollout_batch_size,
    )
    val_build_s = time.perf_counter() - t0
    exhaustive_ms = val_build_s * 1000.0 / max(1, args.n_val_anchors)
    log.info(f"built val imagination labels in {val_build_s:.2f}s")

    def _build_agent() -> DreamerStyleAgent:
        return DreamerStyleAgent(
            in_channels=train_ds.states.shape[-1],
            action_dim=event_space.action_sequences.shape[-1],
            horizon=args.horizon,
            n_candidates=None if args.no_candidate_prior else event_space.n_candidates,
            n_nodes=train_ds.states.shape[1],
            hidden=args.hidden,
            dropout=args.dropout,
        ).to(device)

    val_states = val_ds.states.to(device)
    val_actions = event_space.action_sequences.to(device)
    val_context = val_ds.context_actions.to(device)
    val_risk_norm = val_ds.risk_norm.to(device)

    def _init_pearson(model: DreamerStyleAgent) -> float:
        model.eval()
        with torch.no_grad():
            out = model(val_states, val_actions, context_actions=val_context)
        v = out.values - out.values.mean(dim=1, keepdim=True)
        r = val_risk_norm - val_risk_norm.mean(dim=1, keepdim=True)
        num = (v * r).sum(dim=1)
        denom = (v.pow(2).sum(dim=1) * r.pow(2).sum(dim=1)).sqrt().clamp_min(1e-12)
        return float((num / denom).mean().item())

    agent = _build_agent()
    init_p = _init_pearson(agent)
    log.info(f"init_pearson={init_p:+.4f} (target >= {args.init_pearson_min})")
    if args.init_pearson_min > -1.0:
        for retry in range(1, args.init_pearson_max_retries + 1):
            if init_p >= args.init_pearson_min:
                break
            sub_seed = args.seed + retry * 7919
            set_agent_seed(sub_seed)
            agent = _build_agent()
            init_p = _init_pearson(agent)
            log.info(
                f"init guard retry {retry}: sub_seed={sub_seed} init_pearson={init_p:+.4f}"
            )
        if init_p < args.init_pearson_min:
            log.warning(
                f"init guard failed after {args.init_pearson_max_retries} retries; "
                f"continuing with init_pearson={init_p:+.4f}"
            )
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_score = -float("inf")
    best_payload = {}
    best_path = run_dir / "agent.pt"
    warmup_cfg = dataclass_replace(cfg, actor_loss_weight=0.0, ranking_loss_weight=0.0)
    mid_restarts_used = 0
    for epoch in range(args.epochs):
        epoch_cfg = warmup_cfg if epoch < args.critic_warmup_epochs else cfg
        train_metrics = train_agent_epoch(
            agent,
            train_ds,
            event_space,
            cfg=epoch_cfg,
            device=device,
            optimizer=optimizer,
        )
        eval_result = evaluate_dataset(
            agent,
            val_ds,
            event_space,
            exhaustive_rollout_latency_ms=exhaustive_ms,
        )
        if (
            epoch == args.critic_warmup_epochs - 1
            and args.mid_train_min_pearson > -1.0
            and eval_result.critic_pearson < args.mid_train_min_pearson
            and mid_restarts_used < args.mid_train_max_restarts
        ):
            mid_restarts_used += 1
            log.warning(
                "mid-train guard: critic_pearson=%.4f < %.2f at end of warmup; "
                "re-initializing critic head + value prior (restart %d/%d)",
                eval_result.critic_pearson,
                args.mid_train_min_pearson,
                mid_restarts_used,
                args.mid_train_max_restarts,
            )
            for module in agent.critic.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            if getattr(agent, "value_prior", None) is not None:
                agent.value_prior.data.zero_()
            optimizer = torch.optim.AdamW(
                agent.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
        score = (
            eval_result.critic_pearson
            + eval_result.discover_top10pct_hit_rate
            + eval_result.safe_bottom10pct_hit_rate
        )
        log.info(
            "epoch=%s loss=%.6f critic=%.6f actor=%.6f "
            "pearson=%.4f discover_hit=%.4f safe_hit=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["critic_loss"],
            train_metrics["actor_loss"],
            eval_result.critic_pearson,
            eval_result.discover_top10pct_hit_rate,
            eval_result.safe_bottom10pct_hit_rate,
        )
        payload = {
            "model": agent.state_dict(),
            "agent_cfg": {
                "in_channels": int(train_ds.states.shape[-1]),
                "action_dim": int(event_space.action_sequences.shape[-1]),
                "horizon": args.horizon,
                "n_candidates": None if args.no_candidate_prior else event_space.n_candidates,
                "n_nodes": int(train_ds.states.shape[1]),
                "hidden": args.hidden,
                "dropout": args.dropout,
            },
            "train_cfg": _jsonable_args(args),
            "event_metadata": event_space.metadata,
            "epoch": epoch,
            "eval": eval_result.to_dict(),
        }
        if score > best_score:
            best_score = score
            best_payload = payload
            torch.save(payload, best_path)
            _save_json(run_dir / "eval_agent.json", eval_result.to_dict())

    _save_json(
        run_dir / "config.json",
        {"train_cfg": _jsonable_args(args), "best_score": best_score},
    )
    log.info(f"best_score={best_score:.6f} best_path={best_path}")
    print(f"agent_run_dir={run_dir}")
    if best_payload:
        print(json.dumps(best_payload["eval"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
