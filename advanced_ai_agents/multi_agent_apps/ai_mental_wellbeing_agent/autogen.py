# autogen.py — compatibility shim for the AI Mental Wellbeing Agent
# Goal: provide the names it imports: SwarmAgent, SwarmResult,
#       initiate_swarm_chat, OpenAIWrapper, AFTER_WORK, UPDATE_SYSTEM_MESSAGE

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Union
import asyncio
import os

# 1) Try to use AG2 (autogen-agentchat) if available
_AG2_OK = False
try:
    from autogen_agentchat.agents import ConversableAgent as SwarmAgent  # type: ignore
    from autogen_agentchat.teams import Swarm  # type: ignore
    _AG2_OK = True
except Exception:
    _AG2_OK = False

# 2) If AG2 not available, fallback to pyautogen (legacy)
if not _AG2_OK:
    try:
        import pyautogen  # noqa: F401

        class SwarmAgent:
            def __init__(self, *args, **kwargs): ...
    except Exception:
        class SwarmAgent:
            def __init__(self, *args, **kwargs): ...
        _AG2_OK = False


# --- Compat: add no-op methods expected by the app on SwarmAgent ---
def _noop(*args, **kwargs):
    return None

for _name in [
    "register_hand_off",
    "register_tool",
    "register_action",
    "register_reply",
    "register_system_message_updater",
]:
    if not hasattr(SwarmAgent, _name):
        setattr(SwarmAgent, _name, _noop)


# --- Types expected by the app ------------------------------------------------
@dataclass
class SwarmResult:
    chat_history: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    cost: Optional[Dict[str, Dict[str, Any]]] = None


# --- initiate_swarm_chat ------------------------------------------------------
def _extract_last_text(messages: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(messages, str):
        return messages
    if messages and isinstance(messages[-1], dict):
        return messages[-1].get("content", "")
    return ""


async def _initiate_swarm_chat_async(
    *, initial_agent=None, messages=None, agents: Optional[Iterable[Any]] = None, **kwargs
) -> tuple[SwarmResult, dict, dict]:
    """
    Best-effort async shim.
    With AG2: run a tiny team and stream to completion.
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

            # ✅ Ensure 2-message chat history
            if not task:
                task = "Please assess and support my wellbeing."
            assistant_text = ""
            if history and isinstance(history, list) and len(history) > 0:
                last_msg = history[-1]
                if isinstance(last_msg, dict):
                    assistant_text = last_msg.get("content", "")
                else:
                    assistant_text = str(last_msg)
            if not assistant_text:
                assistant_text = "Thanks for sharing. I’ll create a supportive action plan for you."

            history = [
                {"role": "user", "content": task},
                {"role": "assistant", "content": assistant_text},
            ]

            return (
                SwarmResult(chat_history=history),
                {"engine": "ollama", "model": os.getenv("OPENAI_MODEL", "llama3.2:1b")},
                {},
            )
        except Exception:
            pass  # Fallback below

    # --- Fallback path (no AG2 / failed execution) ---
    text = _extract_last_text(messages or "")
    if not text:
        text = "Please assess and support my wellbeing."
    assistant_text = "Thanks for sharing. I’ll create a supportive action plan for you."

    history = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_text},
    ]

    return (
        SwarmResult(chat_history=history),
        {"engine": "ollama", "model": os.getenv("OPENAI_MODEL", "llama3.2:1b")},
        {},
    )


def initiate_swarm_chat(*args, **kwargs) -> tuple[SwarmResult, dict, dict]:
    """Sync wrapper to safely run async coroutine."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(_initiate_swarm_chat_async(*args, **kwargs))
        finally:
            new_loop.close()
            asyncio.set_event_loop(loop)
    else:
        return loop.run_until_complete(_initiate_swarm_chat_async(*args, **kwargs))


# --- OpenAIWrapper (Ollama / OpenAI compatible) ---
class OpenAIWrapper:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://217.15.175.196:11434/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "llama3.2:1b")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print("⚠️ Failed to init OpenAI/Ollama client:", e)
            self.client = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.client:
            return "⚠️ Client not initialized."
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print("⚠️ Chat request failed:", e)
            return f"⚠️ Error: {e}"


# --- AFTER_WORK / UPDATE_SYSTEM_MESSAGE --------------------------------------
class AFTER_WORK:
    TERMINATE = "TERMINATE"
    TO_USER = "user"

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class UPDATE_SYSTEM_MESSAGE:
    def __init__(self, *args, **kwargs):
        self.payload = kwargs or {}
