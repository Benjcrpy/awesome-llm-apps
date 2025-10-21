# autogen.py — compatibility shim for the AI Mental Wellbeing Agent
# Goal: provide the names it imports: SwarmAgent, SwarmResult,
#       initiate_swarm_chat, OpenAIWrapper, AFTER_WORK, UPDATE_SYSTEM_MESSAGE

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Union

# 1) Try to use AG2 (autogen-agentchat) if available
_AG2_OK = False
try:
    from autogen_agentchat.agents import ConversableAgent as SwarmAgent  # type: ignore
    from autogen_agentchat.teams import Swarm  # type: ignore
    _AG2_OK = True
except Exception:
    _AG2_OK = False

# 2) If AG2 not available, fallback to pyautogen (legacy) just to avoid import crashes
if not _AG2_OK:
    try:
        # pyautogen exposes top-level autogen module traditionally
        # We still define SwarmAgent as a trivial placeholder so import succeeds.
        import pyautogen  # noqa: F401

        class SwarmAgent:  # minimal placeholder
            def __init__(self, *args, **kwargs): ...
    except Exception as _e:
        # As a last resort, provide a stub so the import line doesn't crash.
        class SwarmAgent:  # minimal placeholder
            def __init__(self, *args, **kwargs): ...
        _AG2_OK = False

# --- Types expected by the app ------------------------------------------------
@dataclass
class SwarmResult:
    chat_history: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    cost: Optional[Dict[str, Dict[str, Any]]] = None


# --- initiate_swarm_chat ------------------------------------------------------
# Provide a best-effort implementation when AG2 is present.
# If not, we still return a SwarmResult so the UI can continue.
def _extract_last_text(messages: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(messages, str):
        return messages
    if messages and isinstance(messages[-1], dict):
        return messages[-1].get("content", "")
    return ""

async def initiate_swarm_chat(*, initial_agent=None, messages=None, agents: Optional[Iterable[Any]] = None, **kwargs) -> SwarmResult:  # noqa: D401
    """
    Best-effort async shim. With AG2: run a tiny team and stream to completion.
    Without AG2: just echo the last user message into the result.
    """
    if _AG2_OK and initial_agent is not None:
        try:
            team = Swarm([initial_agent, *(agents or [])])
            task = _extract_last_text(messages or "")
            last = None
            async for last in team.run_stream(task=task):
                pass
            history = getattr(last, "messages", None)
            return SwarmResult(chat_history=history)
        except Exception:
            # Fall back silently to a minimal result
            pass

    # Fallback path (no AG2 or error during run)
    text = _extract_last_text(messages or "")
    return SwarmResult(chat_history=[{"role": "user", "content": text}] if text else None)


# --- OpenAIWrapper ------------------------------------------------------------
# Minimal wrapper so imports succeed; used only if the app calls it.
class OpenAIWrapper:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=api_key)
        except Exception:
            self.client = None
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.client is None:
            return ""
        try:
            # OpenAI>=1.x client
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
            return resp.choices[0].message.content or ""
        except Exception:
            return ""


# --- AFTER_WORK / UPDATE_SYSTEM_MESSAGE --------------------------------------
# Provide lightweight stand-ins so the import line works regardless of version.
class AFTER_WORK:
    TERMINATE = "TERMINATE"
    TO_USER = "user"

class UPDATE_SYSTEM_MESSAGE:
    def __init__(self, *args, **kwargs):
        self.payload = kwargs or {}
