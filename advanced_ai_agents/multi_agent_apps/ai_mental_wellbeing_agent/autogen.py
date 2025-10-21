# autogen.py — delegating shim
# Prefer the real 'autogen' package if installed.
try:
    from importlib import import_module
    _real = import_module("autogen")     # the actual package
    # Re-export everything from the real package
    from autogen import *                # noqa: F401,F403
except Exception:
    # Fallback: try AG2-compatible packages
    try:
        # Most names the app imports can be satisfied by these:
        from autogen_agentchat.agents import ConversableAgent as SwarmAgent, UserProxyAgent
        from autogen_agentchat.teams import Swarm
        # SwarmResult is mostly used as a type; provide a light stub
        from dataclasses import dataclass
        from typing import Any, List, Dict, Optional

        @dataclass
        class SwarmResult:
            chat_history: Optional[List[Dict[str, Any]]] = None
            summary: Optional[str] = None
            cost: Optional[Dict[str, Dict[str, Any]]] = None

        # Try to import swarm helpers if available
        try:
            # Newer AG2 provides initiate_swarm_chat at top-level package;
            # if not available, provide a thin wrapper around teams.Swarm.
            from autogen_agentchat import initiate_swarm_chat  # type: ignore
        except Exception:
            async def initiate_swarm_chat(*, initial_agent, messages, agents, **kwargs):
                # Minimal async shim using the Swarm team (best-effort)
                team = Swarm([initial_agent, *agents])
                stream = team.run_stream(task=messages if isinstance(messages, str) else messages[-1]["content"])
                last = None
                async for last in stream:
                    pass
                return SwarmResult(chat_history=getattr(last, "messages", None))

        # UPDATE_SYSTEM_MESSAGE / AFTER_WORK shims (names changed in newer AG2)
        try:
            from autogen_agentchat.conversable_agent import UPDATE_SYSTEM_MESSAGE  # type: ignore
        except Exception:
            class UPDATE_SYSTEM_MESSAGE:  # fallback no-op marker
                def __init__(self, *args, **kwargs): pass

        try:
            from autogen_agentchat import AfterWorkOption as AFTER_WORK  # type: ignore
        except Exception:
            class AFTER_WORK:  # fallback enum-like
                TERMINATE = "TERMINATE"
                TO_USER = "user"
    except Exception as e2:
        raise ImportError(
            "autogen shim failed: real 'autogen' not installed and AG2 fallbacks unavailable"
        ) from e2
