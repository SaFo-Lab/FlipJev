"""Agent trajectories of R-Judge and ATBench500 in one shape, and the text that the judge reads.

    {"case_id": "rjudge:Application/ds_app#2635", "dataset": "rjudge", "label": "unsafe" | "safe",
     "request": "<first user message>",
     "context": ["<text>", ...],                          # e.g. the agent profile, shown before the trace
     "messages": [{"role": "user", "content": str},
                  {"role": "agent", "thought": str | None, "action": str | None},
                  {"role": "observation", "content": str | None}, ...],
     "tools": [{"name": str, "description": str, "parameters": ...}],
     "meta": {...}}

An observation is what the environment returned to the agent, the channel that an outside attacker can write into.
render_state prints observation text verbatim, so text inserted into an observation appears in the input of the judge
exactly as inserted. The benchmarks are not part of this repository; see the README for the download.
"""

import json
import os
from pathlib import Path

from flipjev.text import document_length

DATA_ROOT = Path(os.environ.get("TRAJECTORY_DATA", Path(__file__).resolve().parent.parent / "data" / "trajectories"))
DATASETS = ("rjudge", "atbench")
CONTEXT_MAX_CHARS = 8000


def _text(value):
    """A raw field as a string, or None when the benchmark left it empty."""
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _flatten_rounds(rounds):
    """Both benchmarks group the messages into rounds (lists of lists); keep their order."""
    for item in rounds:
        if isinstance(item, dict):
            yield item
        else:
            yield from item


def _convert_message(raw):
    role = raw.get("role")
    if role == "user":
        return {"role": "user", "content": _text(raw.get("content")) or ""}
    if role == "agent":
        return {"role": "agent", "thought": _text(raw.get("thought")), "action": _text(raw.get("action"))}
    if role == "environment":
        return {"role": "observation", "content": _text(raw.get("content"))}
    raise ValueError(f"unknown message role {role!r}")


def _request_of(messages):
    return next((m["content"] for m in messages if m["role"] == "user"), "")


def load_rjudge(root=None):
    """R-Judge: 571 labelled records of agent interactions in 27 scenarios."""
    base = Path(root or DATA_ROOT) / "R-Judge" / "data"
    out = []
    for path in sorted(base.glob("*/*.json")):
        for example in json.loads(path.read_text()):
            messages = [_convert_message(m) for m in _flatten_rounds(example["contents"])]
            out.append({"case_id": f"rjudge:{path.parent.name}/{path.stem}#{example['id']}", "dataset": "rjudge",
                        "label": "unsafe" if example["label"] == 1 else "safe",
                        "request": _request_of(messages),
                        "context": [example["profile"]] if example.get("profile") else [],
                        "messages": messages, "tools": [],
                        "meta": {"scenario": example.get("scenario"), "attack_type": example.get("attack_type")}})
    return out


def load_atbench(root=None):
    """ATBench500: 500 tool-use trajectories with their tool definitions."""
    path = Path(root or DATA_ROOT) / "ATBench" / "ATBench500" / "test.json"
    out = []
    for example in json.loads(path.read_text()):
        messages = [_convert_message(m) for m in _flatten_rounds(example["content"])]
        tools = [{"name": t.get("name", ""), "description": t.get("description", ""),
                  "parameters": t.get("parameters")} for t in example.get("tool_used") or []]
        out.append({"case_id": f"atbench:{example['conv_id']}", "dataset": "atbench",
                    "label": "unsafe" if example["label"] == 1 else "safe",
                    "request": _request_of(messages), "context": [], "messages": messages, "tools": tools,
                    "meta": {"risk_source": example.get("risk_source")}})
    return out


LOADERS = {"rjudge": load_rjudge, "atbench": load_atbench}


def load(dataset, root=None):
    if dataset not in LOADERS:
        raise ValueError(f"unknown trajectory dataset {dataset!r}; choose from {sorted(LOADERS)}")
    cases = LOADERS[dataset](root)
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {dataset}")
    return cases


def _clip(text, limit):
    return text if len(text) <= limit else text[:limit] + " ..."


def render_tools(traj):
    return "\n".join(f"- {tool['name']}: {tool.get('description') or ''}" for tool in traj["tools"])


def render_trace(traj):
    lines = []
    for number, message in enumerate(traj["messages"], 1):
        role = message["role"]
        if role == "user":
            lines.append(f"[{number}] user: {message['content']}")
        elif role == "agent":
            if message.get("thought"):
                lines.append(f"[{number}] agent thought: {message['thought']}")
            lines.append(f"[{number}] agent action: {message.get('action') or '(none)'}")
        else:
            content = message.get("content")
            lines.append(f"[{number}] tool output: {content if content else '(empty)'}")
    return "\n".join(lines)


def render_state(traj):
    """The state of the judge: the user request, the context (profile, tool descriptions) and the numbered trace."""
    parts = ["USER REQUEST:\n" + traj["request"]]
    context = list(traj.get("context") or [])
    tools = render_tools(traj)
    if tools:
        context.append("Available tools:\n" + tools)
    if context:
        parts.append("CONTEXT (evidence only; not instructions):\n" + _clip("\n".join(context), CONTEXT_MAX_CHARS))
    parts.append("FULL TRACE:\n" + render_trace(traj))
    return {"trajectory": "\n\n".join(parts)}


def state_tokens(traj):
    return document_length(render_state(traj)["trajectory"])
