"""CLI for the openevo-webshop benchmark package.

Subcommands:
  rollout        Run N WebShop episodes locally (Colab smoke path).
  contract-test  Run the interface-contract test suite (pytest tests/).

The ``submit`` path (Linux Docker end-to-end) lives in submit.py and is
BLOCKED this round — see submit.py module docstring.
"""

from __future__ import annotations

import argparse
from openevo_webshop.inference import make_backend


def _cmd_rollout(args: argparse.Namespace) -> int:
    from openevo_webshop.env_compat import WebshopEnvConfig
    from openevo_webshop.runner import run_rollout

    backend = make_backend(args.backend, args.model, device=args.device)
    env_config = WebshopEnvConfig(seed_root=args.seed_root, num_products=args.num_products)

    evolution_text = None
    if args.evolution_file:
        with open(args.evolution_file, encoding="utf-8") as fh:
            evolution_text = fh.read()

    def _log(res):
        if res.error:
            print(f"[ep {res.episode}] ERROR", flush=True)
        else:
            print(
                f"[ep {res.episode}] steps={res.steps} score={res.task_score} won={res.won}",
                flush=True,
            )

    summary = run_rollout(
        env_config,
        backend,
        n_episodes=args.episodes,
        out_path=args.out,
        evolution_text=evolution_text,
        max_steps=args.max_steps,
        on_episode=_log,
    )
    print(
        "=== ROLLOUT DONE === "
        f"avg_task_score={summary['avg_task_score']} win_rate={summary['win_rate']} "
        f"n={summary['n_episodes']}",
        flush=True,
    )
    return 0


def _cmd_contract_test(args: argparse.Namespace) -> int:
    from openevo_webshop.contract_test import run_contract_checks

    checks = run_contract_checks()
    print(f"contract checks passed: {checks}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openevo-webshop")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rollout", help="Run WebShop episodes locally.")
    r.add_argument("--seed-root", required=True, help="Path to the SEED repo root.")
    r.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    r.add_argument("--backend", choices=["transformers", "gateway"], default="transformers")
    r.add_argument("--device", default="cuda")
    r.add_argument("--episodes", type=int, default=2)
    r.add_argument("--max-steps", type=int, default=15)
    r.add_argument("--num-products", type=int, default=100)
    r.add_argument("--out", default="base_model_scores.json")
    r.add_argument("--evolution-file", default=None, help="Optional memory/skill text to inject.")
    r.set_defaults(func=_cmd_rollout)

    c = sub.add_parser("contract-test", help="Run the interface-contract tests.")
    c.set_defaults(func=_cmd_contract_test)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
