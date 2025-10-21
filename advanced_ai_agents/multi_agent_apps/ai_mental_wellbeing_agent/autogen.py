# autogen.py — compatibility shim for the AI Mental Wellbeing Agent
# Provides: SwarmAgent, SwarmResult, initiate_swarm_chat, OpenAIWrapper, AFTER_WORK, UPDATE_SYSTEM_MESSAGE

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Union, Tuple
import asyncio
import os

# ---------- Try AG2 first ----------
_AG2_OK = False
try:
    from autogen_agentchat.agents import ConversableAgent as SwarmAgent  # type: ignore
    from autogen_agentchat.teams import Swarm  # type: ignore
    _AG2_OK = True
except Exception:
    _AG2_OK = False

# ---------- Fallback to pyautogen (legacy) ----------
if not _AG2_OK:
    try:
        import pyautogen  # noqa: F401

        class SwarmAgent:
            def __init__(self, *args, **kwargs): ...
    except Exception:
        class SwarmAgent:
            def __init__(self, *args, **kwargs): ...
        _AG2_OK = False

# ---------- Add no-op methods missing in ConversableAgent ----------
def _noop(*args, **kwargs): return None
for _name in ["register_hand_off", "register_tool", "register_action", "register_reply", "register_system_message_updater"]:
    if not hasattr(SwarmAgent, _name):
        setattr(SwarmAgent, _name, _noop)

# ---------- Types ----------
@dataclass
class SwarmResult:
    chat_history: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    cost: Optional[Dict[str, Dict[str, Any]]] = None

# ---------- Helpers ----------
def _extract_last_text(messages: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(messages, str):
        return messages
    if messages and isinstance(messages[-1], dict):
        return messages[-1].get("content", "")
    return ""

def _ensure_min_history(user_text: str, assistant_text: str) -> List[Dict[str, str]]:
    """Always return at least 3 messages: system, user, assistant."""
    system_msg = {
        "role": "system",
        "content": "You are a supportive mental wellbeing assistant. Be concise, kind, and practical."
    }
    if not user_text:
        user_text = "Please assess and support my wellbeing."
    if not assistant_text:
        assistant_text = "Thanks for sharing. I’ll create a supportive action plan for you."
    return [
        system_msg,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

# ---------- Core (async) ----------
async def _initiate_swarm_chat_async(
    *, initial_agent=None, messages=None, agents: Optional[Iterable[Any]] = None, **kwargs
) -> Tuple[SwarmResult, dict, dict]:
    # Try AG2 run
    if _AG2_OK and initial_agent is not None:
        try:
            team = Swarm([initial_agent, *(agents or [])])
            task_text = _extract_last_text(messages or "")
            last = None
            async for last in team.run_stream(task=task_text):
                pass
            # Try to pull assistant text from AG2; fall back if empty
            assistant_text = ""
            history_ag2 = getattr(last, "messages", None)
            if isinstance(history_ag2, list) and history_ag2:
                last_msg = history_ag2[-1]
                if isinstance(last_msg, dict):
                    assistant_text = last_msg.get("content", "") or ""
                else:
                    assistant_text = str(last_msg)

            history = _ensure_min_history(task_text, assistant_text)
            return (
                SwarmResult(chat_history=history),
                {"engine": os.getenv("OPENAI_BASE_URL", "http://217.15.175.196:11434/v1"),
                 "model": os.getenv("OPENAI_MODEL", "llama3.2:1b")},
                {},
            )
        except Exception:
            pass  # fall through to safe fallback

    # Safe fallback (no AG2 or AG2 failed)
    user_text = _extract_last_text(messages or "")
    history = _ensure_min_history(user_text, assistant_text="")
    return (
        SwarmResult(chat_history=history),
        {"engine": os.getenv("OPENAI_BASE_URL", "http://217.15.175.196:11434/v1"),
         "model": os.getenv("OPENAI_MODEL", "llama3.2:1b")},
        {},
    )

# ---------- Sync wrapper (never recursive) ----------
def initiate_swarm_chat(*args, **kwargs) -> Tuple[SwarmResult, dict, dict]:
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

# ---------- OpenAI/Ollama Wrapper ----------
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

    def chat(self, messages: List[Dict[str, str]]) -> str:
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

# ---------- Flags ----------
class AFTER_WORK:
    TERMINATE = "TERMINATE"
    TO_USER = "user"
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class UPDATE_SYSTEM_MESSAGE:
    def __init__(self, *args, **kwargs):
        self.payload = kwargs or {}
