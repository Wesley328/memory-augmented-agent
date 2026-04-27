from __future__ import annotations

from dotenv import load_dotenv

from core.llm import LLMError
from service.agent_service import MemoryAgentService

load_dotenv()


def _print_memories(service: MemoryAgentService) -> None:
    print("---- Stored Memories ----")
    memories = service.list_memories(limit=20)
    if not memories:
        print("(empty)")
    for idx, memory in enumerate(memories, start=1):
        status = memory.metadata.get("status", "active")
        version = memory.metadata.get("version", 1)
        print(
            f"{idx}. [{memory.type}] importance={memory.importance:.2f} "
            f"confidence={memory.confidence:.2f} status={status} v{version} "
            f"at {memory.timestamp.isoformat()} -> {memory.content}"
        )
    print("-------------------------")


def main() -> None:
    service = MemoryAgentService()
    print("Memory-Augmented Agent (Simplified)")
    print("Type ':quit' to exit, 'memories' to inspect memory.")
    if not service.llm.is_ready and service.llm.configuration_issue:
        print(f"[Startup] {service.llm.configuration_issue}")

    try:
        while True:
            user_query = input("\nUser: ").strip()
            if not user_query:
                continue
            if user_query.lower() in {":quit", "quit", "exit"}:
                print("Goodbye.")
                break
            if user_query.lower() in {":memories", "/memories", "memories"}:
                _print_memories(service)
                continue

            try:
                result = service.chat(user_query)
            except LLMError as exc:
                print(f"Assistant: {exc.user_message}")
                continue
            except ValueError as exc:
                print(f"Assistant: {exc}")
                continue

            print(f"Assistant: {result.answer}")
            print(
                f"[Planner] action={result.turn_plan.action} "
                f"reason={result.turn_plan.reason}"
            )
            if result.self_check is not None:
                print(f"[Self-check] {result.self_check.summary()}")
            if result.tool_context is not None:
                print(f"[Tool] search -> {result.tool_context}")
            if result.memory_lifecycle is not None:
                lifecycle = result.memory_lifecycle
                if lifecycle.extraction_error is not None:
                    print(
                        "[Memory lifecycle] "
                        f"skipped extraction: {lifecycle.extraction_error}"
                    )
                else:
                    print(
                        "[Memory lifecycle] "
                        f"extracted={lifecycle.extracted} added={lifecycle.added} "
                        f"updated={lifecycle.updated} versioned={lifecycle.versioned} "
                        f"removed={lifecycle.removed}"
                    )
    finally:
        service.close()


if __name__ == "__main__":
    main()
