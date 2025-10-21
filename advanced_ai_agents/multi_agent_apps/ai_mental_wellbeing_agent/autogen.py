# autogen.py — compatibility shim (AG2)
# Ginagawang valid ang `from autogen import ...` kahit naka-install ay autogen-agentchat/pyautogen.

try:
    # Pinaka-wide na re-export: karamihan ng symbols nasa autogen_agentchat
    from autogen_agentchat import *  # noqa: F401,F403
except Exception as e:
    # Fallback sa pyautogen kung mas luma ang packages sa image
    try:
        from pyautogen import *  # noqa: F401,F403
    except Exception as e2:
        # Clear error na madaling makita sa logs kung sakaling pumalya pareho
        raise ImportError(
            "autogen shim failed: neither autogen_agentchat nor pyautogen is importable"
        ) from e2
