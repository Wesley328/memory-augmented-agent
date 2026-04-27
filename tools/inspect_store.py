from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.store import MemoryStore, create_memory_store


def _serialize_memories(memories) -> List[Dict[str, Any]]:
    return [memory.to_dict() for memory in memories]


def _print_table(memories) -> None:
    if not memories:
        print("(empty)")
        return

    for idx, memory in enumerate(memories, start=1):
        metadata = memory.metadata
        print(
            f"{idx}. [{memory.type}] status={metadata.get('status', 'active')} "
            f"v{metadata.get('version', 1)} confidence={memory.confidence:.2f} "
            f"importance={memory.importance:.2f}"
        )
        print(f"   time: {memory.timestamp.isoformat()}")
        print(f"   topic_key: {metadata.get('topic_key', '-')}")
        if metadata.get("lineage_id"):
            print(f"   lineage_id: {metadata.get('lineage_id')}")
        if metadata.get("summary_scope"):
            print(f"   summary_scope: {metadata.get('summary_scope')}")
        print(f"   content: {memory.content}")


def _build_store(args: argparse.Namespace) -> MemoryStore:
    return create_memory_store(
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        max_size=args.max_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or export stored memories from the configured memory store."
    )
    parser.add_argument(
        "--backend",
        default="sqlite",
        help="Store backend to inspect: memory or sqlite. Default is sqlite.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="data/memory.db",
        help="SQLite DB path when backend=sqlite.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="Only used when backend=memory; ignored by persisted data already on disk.",
    )
    parser.add_argument("--status", default=None, help="Filter by memory status.")
    parser.add_argument("--type", dest="memory_type", default=None, help="Filter by memory type.")
    parser.add_argument("--lineage-id", default=None, help="Show a specific version lineage.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary memories.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only show non-summary memories.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of memories to show.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path for JSON export.",
    )
    args = parser.parse_args()

    if args.summary_only and args.base_only:
        raise SystemExit("Use only one of --summary-only or --base-only.")

    store = _build_store(args)
    try:
        summary_only = True if args.summary_only else False if args.base_only else None
        if args.lineage_id:
            memories = store.get_lineage(args.lineage_id)
            if args.limit is not None:
                memories = memories[: args.limit]
        else:
            memories = store.query_memories(
                status=args.status,
                memory_type=args.memory_type,
                summary_only=summary_only,
                limit=args.limit,
            )

        if args.json or args.output:
            payload = _serialize_memories(memories)
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(rendered + "\n")
            if args.json:
                print(rendered)
            elif args.output:
                print(f"Wrote {len(payload)} memories to {args.output}")
        else:
            _print_table(memories)
    finally:
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
