"""Replay one recorded, read-only Gaia tool call outside the chat runtime.

Run from the repository root with a verification step on standard input::

    python -m src.gaia_agent.replay < call.json
"""

from __future__ import annotations

import asyncio
import json
import sys

from src.gaia_agent.runtime.contracts import ToolExecutionError
from src.gaia_agent.tools.registry import GaiaToolset


REPLAYABLE_CAPABILITIES = frozenset({
    "retrieve_schema",
    "search_darwin_core_terms",
    "get_darwin_core_terms",
    "query_occurrences",
    "query_statistics",
})


def parse_call(value: object) -> tuple[str, dict[str, object]]:
    """Extract a capability and arguments from one public verification step."""
    if not isinstance(value, dict):
        raise ValueError("Input must be one JSON object.")
    capability = value.get("capability")
    arguments = value.get("arguments")
    if not isinstance(capability, str) or capability not in REPLAYABLE_CAPABILITIES:
        raise ValueError(
            "Capability must be a supported read-only Gaia tool: "
            + ", ".join(sorted(REPLAYABLE_CAPABILITIES))
            + "."
        )
    if not isinstance(arguments, dict):
        raise ValueError("The call must contain an arguments JSON object.")
    return capability, arguments


async def replay_call(capability: str, arguments: dict[str, object]) -> object:
    """Execute the registered handler and return its model-visible observation."""
    registered = {
        tool.definition.name: tool
        for tool in GaiaToolset.load().registered_tools()
    }
    tool = registered[capability]
    result = await tool.handler(arguments.copy())
    if tool.observation_projector is not None:
        result = tool.observation_projector(result)
    return result


def main() -> int:
    """Read a verification step from stdin and print its replayed result."""
    try:
        capability, arguments = parse_call(json.load(sys.stdin))
        result = asyncio.run(replay_call(capability, arguments))
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Invalid replay input: {error}", file=sys.stderr)
        return 2
    except ToolExecutionError as error:
        print(f"Gaia call failed ({error.code}): {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Gaia call could not be completed. Check this instance's database and configuration.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
