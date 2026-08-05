"""
Pytest suite for the tool-calling agent loop, with the model fully mocked.

Per exercise3/CLAUDE.md: extraction-stage tests use fixed/recorded model
outputs rather than live API calls, so the suite stays deterministic and fast.
No network or Ollama server is required to run this file.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent
from agent import HER2AgentError, run_her2_agent, run_her2_agent_verbose


def make_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def make_response(content: str = None, tool_calls: list = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def mock_create(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(agent.client.chat.completions, "create", mock)
    return mock


def test_happy_path_tool_call_first_try(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "3+"}))]
        ),
        make_response(content="This report indicates HER2 Positive status."),
    ]

    result = run_her2_agent("Invasive ductal carcinoma. HER2 IHC 3+.")

    assert "HER2 Positive" in result
    assert agent.DISCLAIMER in result
    assert mock_create.call_count == 2


def test_retry_then_success(mock_create):
    mock_create.side_effect = [
        make_response(content="I think this looks positive."),  # no tool call
        make_response(
            tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "0"}))]
        ),
        make_response(content="This report indicates HER2 Negative status."),
    ]

    result = run_her2_agent("HER2 IHC 0.")

    assert "HER2 Negative" in result
    assert agent.DISCLAIMER in result
    assert mock_create.call_count == 3


def test_no_tool_call_after_retry_raises(mock_create):
    mock_create.side_effect = [
        make_response(content="prose answer one"),
        make_response(content="prose answer two"),
    ]

    with pytest.raises(HER2AgentError):
        run_her2_agent("HER2 IHC 2+, no ISH mentioned.")

    assert mock_create.call_count == 2 # signals 2 calls to the model: first attempt, then retry after reminder (failure condition)


def test_unknown_tool_name_raises(mock_create):
    mock_create.side_effect = [
        make_response(tool_calls=[make_tool_call("call_1", "some_other_tool", "{}")]),
    ]

    with pytest.raises(HER2AgentError):
        run_her2_agent("HER2 IHC 3+.")


def test_malformed_json_arguments_raises(mock_create):
    mock_create.side_effect = [
        make_response(tool_calls=[make_tool_call("call_1", "score_her2_status", "{not valid json")]),
    ]

    with pytest.raises(HER2AgentError):
        run_her2_agent("HER2 IHC 3+.")


def test_non_object_json_arguments_raises(mock_create):
    mock_create.side_effect = [
        make_response(tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps("3+"))]),
    ]

    with pytest.raises(HER2AgentError):
        run_her2_agent("HER2 IHC 3+.")


def test_invalid_field_value_raises(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "4+"}))]
        ),
    ]

    with pytest.raises(HER2AgentError) as exc_info:
        run_her2_agent("HER2 IHC 4+ (garbled report).")
    assert "4+" in str(exc_info.value)


def test_unexpected_argument_key_raises(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[
                make_tool_call(
                    "call_1",
                    "score_her2_status",
                    json.dumps({"ihc_score": "3+", "notes": "unexpected extra field"}),
                )
            ]
        ),
    ]

    with pytest.raises(HER2AgentError):
        run_her2_agent("HER2 IHC 3+.")


def test_connection_failure_raises(mock_create):
    mock_create.side_effect = ConnectionError("Connection refused")

    with pytest.raises(HER2AgentError) as exc_info:
        run_her2_agent("HER2 IHC 3+.")
    assert "ollama" in str(exc_info.value).lower()


def test_multiple_tool_calls_uses_first_only(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[
                make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "0"})),
                make_tool_call("call_2", "score_her2_status", json.dumps({"ihc_score": "3+"})),
            ]
        ),
        make_response(content="Final answer."),
    ]

    run_her2_agent("HER2 IHC ambiguous report.")

    second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    tool_result = json.loads(tool_messages[0]["content"])
    assert tool_result["status"] == "Negative"  # from the first tool call (ihc_score="0")


def test_empty_report_text_raises_value_error(mock_create):
    with pytest.raises(ValueError):
        run_her2_agent("")
    assert mock_create.call_count == 0


def test_whitespace_only_report_text_raises_value_error(mock_create):
    with pytest.raises(ValueError):
        run_her2_agent("   \n\t  ")
    assert mock_create.call_count == 0


def test_none_report_text_raises_value_error(mock_create):
    with pytest.raises(ValueError):
        run_her2_agent(None)
    assert mock_create.call_count == 0


def test_non_string_report_text_raises_value_error(mock_create):
    with pytest.raises(ValueError):
        run_her2_agent(12345)
    assert mock_create.call_count == 0


def test_oversized_report_text_raises_value_error(mock_create):
    with pytest.raises(ValueError):
        run_her2_agent("x" * (agent.MAX_REPORT_CHARS + 1))
    assert mock_create.call_count == 0


def test_verbose_returns_intermediate_data(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "3+"}))]
        ),
        make_response(content="This report indicates HER2 Positive status."),
    ]

    pipeline = run_her2_agent_verbose("HER2 IHC 3+.")

    assert pipeline["tool_arguments"] == {"ihc_score": "3+"}
    assert pipeline["tool_result"]["status"] == "Positive"
    assert pipeline["final_text"] == "This report indicates HER2 Positive status."
    assert agent.DISCLAIMER not in pipeline["final_text"]  # disclaimer is only added by run_her2_agent


def test_final_call_omits_tools_param(mock_create):
    mock_create.side_effect = [
        make_response(
            tool_calls=[make_tool_call("call_1", "score_her2_status", json.dumps({"ihc_score": "3+"}))]
        ),
        make_response(content="Final answer."),
    ]

    run_her2_agent("HER2 IHC 3+.")

    second_call_kwargs = mock_create.call_args_list[1].kwargs
    assert "tools" not in second_call_kwargs or not second_call_kwargs["tools"]
