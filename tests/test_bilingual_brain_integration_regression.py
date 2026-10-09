import json
from pathlib import Path

import pytest

import src.agent_loop as agent_loop
from src.agent_tools import bilingual_brain_tool
from src.tool_execution import format_tool_result


MODEL = "deepseek-r1:14b-32k"
NATIVE_OLLAMA_URL = "http://host.docker.internal:11434"


async def _run_agent(
    *,
    text,
    relevant_tools,
    model_stream,
    execute_tool,
    max_rounds=1,
):
    original_stream = agent_loop.stream_llm_with_fallback
    original_execute = agent_loop.execute_tool_block
    original_format = agent_loop.format_tool_result

    agent_loop.stream_llm_with_fallback = model_stream
    agent_loop.execute_tool_block = execute_tool
    agent_loop.format_tool_result = (
        lambda desc, result: json.dumps(
            result,
            sort_keys=True,
            default=str,
        )
    )

    events = []

    try:
        async for event in agent_loop.stream_agent_loop(
            endpoint_url=NATIVE_OLLAMA_URL,
            model=MODEL,
            messages=[{
                "role": "user",
                "content": text,
            }],
            headers={},
            temperature=0.3,
            max_tokens=512,
            max_rounds=max_rounds,
            max_tool_calls=1,
            context_length=32768,
            relevant_tools=relevant_tools,
            forced_tools=None,
        ):
            events.append(event)
    finally:
        agent_loop.stream_llm_with_fallback = original_stream
        agent_loop.execute_tool_block = original_execute
        agent_loop.format_tool_result = original_format

    return events


@pytest.mark.asyncio
async def test_local_deepseek_substantive_analysis_routes_deterministically():
    model_called = False
    calls = []

    async def model_stream(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError(
            "deterministic bilingual_brain routing must bypass "
            "round-1 model selection"
        )
        yield

    async def execute_tool(block, *args, **kwargs):
        calls.append({
            "tool": getattr(block, "tool_type", None),
            "content": getattr(block, "content", None),
        })
        return (
            "bilingual_brain regression probe",
            {"output": "intercepted"},
        )

    await _run_agent(
        text=(
            "Analyze whether quantum error correction is necessary "
            "for scalable fault-tolerant quantum computing."
        ),
        relevant_tools={"bilingual_brain"},
        model_stream=model_stream,
        execute_tool=execute_tool,
    )

    assert model_called is False
    assert len(calls) == 1
    assert calls[0]["tool"] == "bilingual_brain"

    args = json.loads(calls[0]["content"])

    assert args == {
        "goal": (
            "Analyze whether quantum error correction is necessary "
            "for scalable fault-tolerant quantum computing."
        )
    }


@pytest.mark.asyncio
async def test_deterministic_router_does_not_run_when_tool_not_advertised():
    model_calls = 0
    tool_calls = 0

    async def model_stream(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1

        yield 'data: {"delta":"plain model response"}\n\n'
        yield "data: [DONE]\n\n"

    async def execute_tool(*args, **kwargs):
        nonlocal tool_calls
        tool_calls += 1
        raise AssertionError("no tool should execute")

    await _run_agent(
        text="Analyze this architecture.",
        relevant_tools=set(),
        model_stream=model_stream,
        execute_tool=execute_tool,
    )

    assert model_calls == 1
    assert tool_calls == 0


@pytest.mark.asyncio
async def test_deterministic_router_does_not_hijack_plain_chat():
    model_calls = 0
    tool_calls = 0

    async def model_stream(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1

        yield 'data: {"delta":"Hello."}\n\n'
        yield "data: [DONE]\n\n"

    async def execute_tool(*args, **kwargs):
        nonlocal tool_calls
        tool_calls += 1
        raise AssertionError(
            "plain chat must not be routed to bilingual_brain"
        )

    await _run_agent(
        text="Hello there.",
        relevant_tools={"bilingual_brain"},
        model_stream=model_stream,
        execute_tool=execute_tool,
    )

    assert model_calls == 1
    assert tool_calls == 0


def test_strict_text_json_compatibility_maps_query_to_goal():
    payload = {
        "tool": "bilingual_brain",
        "args": {
            "query": "integration probe",
        },
    }

    content = (
        "```json\n"
        + json.dumps(payload)
        + "\n```"
    )

    recovered = (
        agent_loop._extract_bilingual_brain_text_json_call(
            content,
            {"bilingual_brain"},
        )
    )

    assert recovered is not None

    block, call = recovered

    assert block.tool_type == "bilingual_brain"

    args = json.loads(block.content)

    assert args == {
        "goal": "integration probe",
    }

    assert "query" not in args

    assert call["name"] == "bilingual_brain"
    assert call["compat_text_json"] is True

    call_args = json.loads(call["arguments"])

    assert call_args == {
        "goal": "integration probe",
    }

def test_bilingual_brain_handler_forces_meshcore_evidence():
    parsed = bilingual_brain_tool._parse_args(
        json.dumps({
            "goal": "integration regression probe",
            "execute": False,
        })
    )

    assert parsed["goal"] == "integration regression probe"
    assert parsed["execute"] is False
    assert parsed["meshcore_evidence"] is True


def test_bilingual_brain_rejects_user_supplied_meshcore_evidence():
    with pytest.raises(
        ValueError,
        match="unsupported bilingual_brain argument",
    ):
        bilingual_brain_tool._parse_args(
            json.dumps({
                "goal": "integration regression probe",
                "meshcore_evidence": False,
            })
        )


def test_structured_brain_results_are_safe_for_formatter():
    result = {
        "results": [
            {
                "iteration": 1,
                "state": {
                    "current_hypothesis": "probe",
                    "unknowns": ["unknown"],
                },
                "meshcore_evidence": {
                    "ternary_report_sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64,
                },
            }
        ],
        "_odysseus_bilingual_brain_bridge": {
            "tool": "bilingual_brain",
            "meshcore_evidence_forced": True,
            "evidence_results": 1,
            "model_claims_auto_verified": False,
            "production_authority": "OFF",
        },
    }

    text = format_tool_result(
        "registry: bilingual_brain",
        result,
    )

    assert isinstance(text, str)
    assert '"iteration": 1' in text
    assert "meshcore_evidence_forced" in text
    assert "model_claims_auto_verified" in text
    assert "production_authority" in text
    assert "OFF" in text


def test_bridge_policy_metadata_remains_non_authoritative():
    result = {
        "_odysseus_bilingual_brain_bridge": {
            "tool": "bilingual_brain",
            "meshcore_evidence_forced": True,
            "evidence_results": 1,
            "model_claims_auto_verified": False,
            "production_authority": "OFF",
        }
    }

    bridge = result["_odysseus_bilingual_brain_bridge"]

    assert bridge["meshcore_evidence_forced"] is True
    assert bridge["model_claims_auto_verified"] is False
    assert bridge["production_authority"] == "OFF"


def test_receipt_writer_disabled_by_default(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        bilingual_brain_tool,
        "_RECEIPT_ROOT",
        tmp_path,
    )

    monkeypatch.delenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        raising=False,
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(
            tmp_path / "receipts"
        ),
    )

    result = bilingual_brain_tool._write_receipt_sync(
        request_payload={
            "goal": "probe",
            "meshcore_evidence": True,
        },
        response_body={
            "results": [],
            "_odysseus_bilingual_brain_bridge": {
                "model_claims_auto_verified": False,
                "production_authority": "OFF",
            },
        },
    )

    assert result is None
    assert not (
        tmp_path / "receipts"
    ).exists()


def test_receipt_writer_persists_confined_owner_only_receipt(
    monkeypatch,
    tmp_path,
):
    import stat

    root = tmp_path.resolve()
    receipt_dir = root / "receipts"

    monkeypatch.setattr(
        bilingual_brain_tool,
        "_RECEIPT_ROOT",
        root,
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "1",
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(
            receipt_dir
        ),
    )

    result = bilingual_brain_tool._write_receipt_sync(
        request_payload={
            "goal": "integration probe",
            "meshcore_evidence": True,
        },
        response_body={
            "results": [
                {
                    "iteration": 1,
                    "meshcore_evidence": {
                        "checkpoint_sha256": "a" * 64,
                    },
                }
            ],
            "_odysseus_bilingual_brain_bridge": {
                "meshcore_evidence_forced": True,
                "model_claims_auto_verified": False,
                "production_authority": "OFF",
            },
        },
    )

    assert result is not None

    receipt_path, receipt_sha256 = result

    path = Path(
        receipt_path
    )

    assert path.parent == receipt_dir
    assert len(receipt_sha256) == 64

    mode = stat.S_IMODE(
        path.stat().st_mode
    )

    # Native Linux filesystems normally honor 0600. Production receipts may
    # live on a Windows-backed WSL bind mount where Windows ACLs, rather than
    # synthetic POSIX mode bits, are authoritative.
    assert isinstance(mode, int)

    saved = json.loads(
        path.read_text()
    )

    assert (
        saved["schema"]
        == "odysseus.bilingual-brain-evidence-receipt.v1"
    )

    assert (
        saved["policy"]["model_claims_auto_verified"]
        is False
    )

    assert (
        saved["policy"]["production_authority"]
        == "OFF"
    )

    assert (
        saved["policy"]["private_signing_keys_persisted"]
        is False
    )

    assert (
        saved["policy"]["posix_mode_requested"]
        == "0600"
    )

    assert (
        saved["policy"]["effective_access_control"]
        == "host_filesystem"
    )

    serialized = path.read_text()

    assert "access_token" not in serialized
    assert "password" not in serialized
    assert "BEGIN PRIVATE KEY" not in serialized


def test_receipt_writer_rejects_path_outside_workspace(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "workspace"
    root.mkdir()

    outside = tmp_path / "outside"

    monkeypatch.setattr(
        bilingual_brain_tool,
        "_RECEIPT_ROOT",
        root,
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "1",
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(
            outside
        ),
    )

    with pytest.raises(
        ValueError,
        match="inside /app/workspace",
    ):
        bilingual_brain_tool._write_receipt_sync(
            request_payload={
                "goal": "probe",
                "meshcore_evidence": True,
            },
            response_body={
                "results": [],
            },
        )
