"""
Tool-calling agent for HER2 status determination, backed by the local
Gemma model served through Ollama.

The model's only job is to read a free-text pathology snippet and call the
score_her2_status tool (exercise3/her2_tools.py) with the fields it
extracted — it never computes the HER2 classification itself. That keeps
the actual staging decision deterministic and auditable; see
exercise3/CLAUDE.md.

For educational use only — not a clinical decision tool.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from her2_tools import HER2_SCORE_TOOL, score_her2_status

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
MODEL = os.environ.get("HER2_AGENT_MODEL", "gemma4:e2b")

MAX_REPORT_CHARS = 20_000

DISCLAIMER = "⚠️ For educational use only — not a clinical decision tool. Do not use with real patient data."

SYSTEM_PROMPT = (
    "You are a pathology report field-extraction assistant for an educational exercise. "
    "Given a free-text pathology report snippet, extract the HER2 IHC score and, if present, "
    "the ISH HER2/CEP17 ratio and average HER2 copy number. Call the score_her2_status tool "
    "with those fields — you must never compute or state the HER2 classification yourself; "
    "that decision is made deterministically by the tool. If ISH values are not mentioned, "
    "omit them or pass null."
)

RETRY_REMINDER = (
    "You did not call the score_her2_status tool. You must call it with the extracted "
    "ihc_score (and ish_ratio/ish_copy_number if present) instead of answering in prose."
)

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


class HER2AgentError(Exception):
    """Raised for any tool-calling agent failure (model, network, or tool-use error)."""


def _validate_report_text(report_text: Any) -> str:
    if not isinstance(report_text, str):
        raise ValueError(f"report_text must be a string, got {type(report_text).__name__}")
    cleaned = report_text.strip()
    if not cleaned:
        raise ValueError("report_text must not be empty")
    if len(cleaned) > MAX_REPORT_CHARS:
        raise ValueError(f"report_text exceeds max length of {MAX_REPORT_CHARS} characters")
    return cleaned


def _request_message(messages: List[Dict[str, Any]], with_tools: bool) -> Any:
    kwargs: Dict[str, Any] = {"model": MODEL, "messages": messages}
    if with_tools:
        kwargs["tools"] = [HER2_SCORE_TOOL]
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise HER2AgentError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL} — is it running? ({exc})"
        ) from exc
    return response.choices[0].message


def run_her2_agent_verbose(report_text: str) -> Dict[str, Any]:
    """
    Like run_her2_agent, but also returns the intermediate tool-call
    arguments and the raw score_her2_status result, not just the model's
    final formatted text. Useful for manually checking, against the real
    model, whether the extracted fields and the deterministic classification
    actually agree with each other — see manual_smoke_test.py.

    Returns {"tool_arguments": dict, "tool_result": dict, "final_text": str}.

    Raises ValueError for invalid report_text, or HER2AgentError for any
    model/tool-calling failure (no tool call after retry, unknown tool,
    malformed arguments, invalid extracted field values, or a connection
    failure reaching Ollama).
    """
    cleaned_report = _validate_report_text(report_text)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}, # specifies the task for the agent
        {"role": "user", "content": cleaned_report}, # provides the pathology report snippet to the agent
    ]

    message = _request_message(messages, with_tools=True) # call the model
    tool_calls = getattr(message, "tool_calls", None) # check if the model called any tools

    if not tool_calls: # if the model does not call the tool, retry once with a remainder, then fail
        messages.append({"role": "assistant", "content": message.content or ""})
        messages.append({"role": "user", "content": RETRY_REMINDER})
        message = _request_message(messages, with_tools=True) 
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            raise HER2AgentError(
                f"{MODEL} did not call score_her2_status even after a reminder retry."
            )

    tool_call = tool_calls[0]  # only one tool exists; ignore any extras -> the only tool the model should call is score_her2_status

    if tool_call.function.name != "score_her2_status":
        raise HER2AgentError(f"Model called an unknown tool: {tool_call.function.name!r}")

    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise HER2AgentError(f"Model produced malformed tool-call arguments: {exc}") from exc

    if not isinstance(arguments, dict):
        raise HER2AgentError(
            f"Tool-call arguments must be a JSON object, got {type(arguments).__name__}"
        )

    try:
        result = score_her2_status(**arguments) # ihc_score, ish_ratio, ish_copy_number
    except (ValueError, TypeError) as exc:
        raise HER2AgentError(f"Model extracted invalid fields: {exc}") from exc

    messages.append( # append the tool call and its result to the message history
        {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append( # append the tool result to the message history
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result),
        }
    )

    final_message = _request_message(messages, with_tools=False) # ask the model to summarize the result (message fields defined above) for the user, without calling any tools
    final_text = final_message.content or ""
    return {"tool_arguments": arguments, "tool_result": result, "final_text": final_text}


def run_her2_agent(report_text: str) -> str:
    """
    Run the HER2 tool-calling agent on a free-text pathology snippet and
    return a formatted, disclaimer-carrying result. Thin wrapper around
    run_her2_agent_verbose for callers that only want the final text.
    """
    pipeline = run_her2_agent_verbose(report_text)
    return f"{pipeline['final_text']}\n\n{DISCLAIMER}"
