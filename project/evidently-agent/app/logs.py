"""
logs.py
-------
Utilities for logging agent interactions to JSON files.
"""

import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

LOG_DIR = Path(os.getenv("LOGS_DIRECTORY", "logs"))
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not JSON-serializable")


def _extract_tools(agent) -> list[str]:
    tools = []
    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())
    return tools


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_entry(agent, messages, source: str = "user") -> dict:
    """Build a log dict from an agent run."""
    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)
    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": getattr(agent.model, "system", str(type(agent.model))),
        "model": getattr(agent.model, "model_name", str(agent.model)),
        "tools": _extract_tools(agent),
        "messages": dict_messages,
        "source": source,
    }


def log_interaction_to_file(agent, messages, source: str = "user") -> Path:
    """
    Persist an agent interaction to a timestamped JSON file.

    Returns the path of the written file.
    """
    entry = log_entry(agent, messages, source)

    # Use the timestamp of the last message
    last_ts = entry["messages"][-1].get("timestamp", datetime.utcnow().isoformat())
    if isinstance(last_ts, str):
        ts_obj = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    else:
        ts_obj = last_ts
    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=_serializer)

    return filepath


def load_log_file(log_file) -> dict:
    """Load a JSON log file and attach the filename for tracking."""
    with open(log_file, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    data["log_file"] = Path(log_file)
    return data


def list_log_files(agent_name: str | None = None, source: str | None = None) -> list[Path]:
    """
    List log files, optionally filtered by agent name and/or source.

    Args:
        agent_name: If given, only return files whose name contains this string.
        source:     If given, only return logs with matching 'source' field.
    """
    files = sorted(LOG_DIR.glob("*.json"))
    results = []
    for f in files:
        if agent_name and agent_name not in f.name:
            continue
        if source:
            try:
                data = load_log_file(f)
                if data.get("source") != source:
                    continue
            except Exception:  # noqa: BLE001
                continue
        results.append(f)
    return results
