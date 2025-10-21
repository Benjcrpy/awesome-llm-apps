# autogen.py — compatibility shim (AG2)
try:
    from autogen_agentchat import *  # noqa: F401,F403
except Exception:
    from pyautogen import *  # noqa: F401,F403
