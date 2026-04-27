from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from observability.trace_store import TurnTraceStore, create_turn_trace_store


def _build_store(args: argparse.Namespace) -> TurnTraceStore:
    return create_turn_trace_store(
        backend=args.backend,
        sqlite_path=args.sqlite_path,
    )


def _print_table(items: list[dict[str, Any]]) -> None:
    if not items:
        print("(empty)")
        return

    for item in items:
        turn_plan = item.get("turn_plan") or {}
        self_check = item.get("self_check") or {}
        print(
            f"turn={item.get('turn_id')} route={item.get('query_route')} "
            f"action={turn_plan.get('action', '-')}"
        )
        print(f"  time: {item.get('created_at')}")
        print(f"  query: {item.get('query', '')}")
        print(f"  answer: {item.get('answer', '')}")
        print(f"  planner_reason: {turn_plan.get('reason', '-')}")
        if self_check:
            print(f"  self_check: {self_check.get('summary', '-')}")
        lifecycle = item.get("memory_lifecycle") or {}
        if lifecycle:
            print(
                "  memory_lifecycle: "
                f"extracted={lifecycle.get('extracted', 0)} "
                f"added={lifecycle.get('added', 0)} "
                f"updated={lifecycle.get('updated', 0)} "
                f"versioned={lifecycle.get('versioned', 0)} "
                f"removed={lifecycle.get('removed', 0)}"
            )
        print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect persisted turn traces from the local observability store."
    )
    parser.add_argument(
        "--backend",
        default="sqlite",
        help="Trace store backend to inspect: memory or sqlite. Default is sqlite.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="data/memory.db",
        help="SQLite DB path when backend=sqlite.",
    )
    parser.add_argument("--turn-id", type=int, default=None, help="Inspect a single turn trace.")
    parser.add_argument("--query-route", default=None, help="Filter by query route.")
    parser.add_argument("--planner-action", default=None, help="Filter by planner action.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of traces to show.")
    parser.add_argument("--json", action="store_true", help="Print traces as JSON.")
    args = parser.parse_args()

    store = _build_store(args)
    try:
        if args.turn_id is not None:
            trace = store.get(args.turn_id)
            payload = [] if trace is None else [trace.to_dict()]
        else:
            payload = [
                trace.to_dict()
                for trace in store.list(
                    query_route=args.query_route,
                    planner_action=args.planner_action,
                    limit=args.limit,
                )
            ]

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        _print_table(payload)
    finally:
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
