import inspect

import pytest

import src.agent_loop as agent_loop

from src.agent_loop import (
    _bilingual_brain_receipt_args_from_text,
    _classify_agent_request,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "show my latest Bilingual Brain receipt",
            {
                "action": "latest",
            },
        ),
        (
            "verify my latest evidence receipt",
            {
                "action": "latest",
            },
        ),
        (
            "list all my Bilingual Brain receipts",
            {
                "action": "list",
                "limit": 10,
            },
        ),
        (
            (
                "verify Bilingual Brain receipt "
                "20261008T041341.195158Z-"
                "0d9a1c9b774b79ab.json"
            ),
            {
                "action": "verify",
                "receipt": (
                    "20261008T041341.195158Z-"
                    "0d9a1c9b774b79ab.json"
                ),
            },
        ),
        (
            (
                "read Brain receipt "
                "20261008T041341.195158Z-"
                "0d9a1c9b774b79ab.json"
            ),
            {
                "action": "read",
                "receipt": (
                    "20261008T041341.195158Z-"
                    "0d9a1c9b774b79ab.json"
                ),
            },
        ),
    ],
)
def test_receipt_text_to_tool_args(
    text,
    expected,
):
    assert (
        _bilingual_brain_receipt_args_from_text(
            text
        )
        == expected
    )


@pytest.mark.parametrize(
    "text",
    [
        "show my grocery receipt",
        "what is the latest news",
        "show settings",
        "analyze quantum computing",
    ],
)
def test_non_receipt_requests_are_not_hijacked(
    text,
):
    assert (
        _bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "show my latest Bilingual Brain receipt",
        "verify my latest evidence receipt",
        "list all my Bilingual Brain receipts",
    ],
)
def test_receipt_domain_avoids_web_and_ui(
    text,
):
    intent = _classify_agent_request(
        [],
        text,
    )

    domains = intent["domains"]

    assert "brain_receipts" in domains
    assert "web" not in domains
    assert "ui" not in domains
    assert intent["low_signal"] is False


def test_normal_latest_news_still_routes_web():
    intent = _classify_agent_request(
        [],
        "what is the latest news",
    )

    assert "web" in intent["domains"]
    assert "brain_receipts" not in intent["domains"]


def test_substantive_bilingual_brain_receipt_system_analysis_is_not_receipt_read():
    from src.agent_loop import (
        _bilingual_brain_receipt_args_from_text,
    )

    text = (
        "Analyze with the Bilingual Brain whether the current REVA "
        "evidence-receipt integration preserves a clean separation "
        "between evidence transport and semantic verification. "
        "Explain the structural result in English."
    )

    assert (
        _bilingual_brain_receipt_args_from_text(text)
        is None
    )


def test_explicit_bilingual_brain_analysis_seeds_brain_tool_not_receipt_reader():
    from src.agent_loop import (
        _DOMAIN_TOOL_MAP,
        _bilingual_brain_receipt_args_from_text,
        _classify_agent_request,
    )

    text = (
        "Analyze with the Bilingual Brain whether the current REVA "
        "evidence-receipt integration preserves a clean separation "
        "between evidence transport and semantic verification. "
        "Explain the structural result in English."
    )

    assert (
        _bilingual_brain_receipt_args_from_text(text)
        is None
    )

    intent = _classify_agent_request(
        [],
        text,
    )

    domains = set(
        intent.get("domains")
        or set()
    )

    assert "bilingual_brain" in domains
    assert "brain_receipts" not in domains

    seeded_tools = set()

    for domain in domains:
        seeded_tools.update(
            _DOMAIN_TOOL_MAP.get(
                domain,
                set(),
            )
        )

    assert "bilingual_brain" in seeded_tools
    assert "bilingual_brain_receipts" not in seeded_tools


def test_receipt_latest_request_still_routes_to_receipt_reader():
    from src.agent_loop import (
        _bilingual_brain_receipt_args_from_text,
        _classify_agent_request,
    )

    text = "show my latest Bilingual Brain receipt"

    assert (
        _bilingual_brain_receipt_args_from_text(text)
        == {
            "action": "latest",
        }
    )

    intent = _classify_agent_request(
        [],
        text,
    )

    domains = set(
        intent.get("domains")
        or set()
    )

    assert "brain_receipts" in domains
    assert "bilingual_brain" not in domains


def test_receipt_system_discussion_without_access_intent_does_not_read_receipts():
    from src.agent_loop import (
        _bilingual_brain_receipt_args_from_text,
    )

    examples = [
        (
            "Explain how the Bilingual Brain receipt system "
            "separates transport from semantic verification."
        ),
        (
            "Compare the MeshCore evidence-receipt architecture "
            "with a conventional audit log."
        ),
        (
            "Assess whether evidence receipts preserve provenance "
            "without semantically verifying model claims."
        ),
    ]

    for text in examples:
        assert (
            _bilingual_brain_receipt_args_from_text(text)
            is None
        )


def test_bilingual_brain_domain_has_prompt_rule():
    from src.agent_loop import (
        _DOMAIN_RULES,
        _DOMAIN_TOOL_MAP,
    )

    assert (
        "bilingual_brain"
        in _DOMAIN_TOOL_MAP
    )

    assert (
        "bilingual_brain"
        in _DOMAIN_RULES
    )

    rule = _DOMAIN_RULES[
        "bilingual_brain"
    ]

    assert isinstance(
        rule,
        str,
    )

    assert (
        "PRODUCTION_AUTHORITY"
        in rule
    )

    assert (
        "not automatically verified"
        in rule
    )


def test_bilingual_brain_prompt_rule_assembly_does_not_raise():
    from src.agent_loop import (
        _domain_rules_for_tools,
    )

    rules = (
        _domain_rules_for_tools(
            {
                "bilingual_brain",
            }
        )
    )

    assert rules

    combined = "\n".join(
        str(rule)
        for rule in rules
    )

    assert (
        "Bilingual Brain execution rule"
        in combined
    )


def _sample_receipt_list_result() -> dict:
    return {
        "results": [
            {
                "receipt":
                    "20261009T043350.153555Z-"
                    "3d2b1e1fc81ad28d.json",
                "size_bytes": 4096,
                # These fields deliberately simulate future/untrusted extras.
                "content":
                    "DO_NOT_RENDER_RECEIPT_CONTENT",
                "payload":
                    "DO_NOT_RENDER_RECEIPT_PAYLOAD",
                "detail":
                    "DO_NOT_RENDER_RECEIPT_DETAIL",
            },
            {
                "receipt":
                    "20261009T003543.155051Z-"
                    "9ff0b4221c44f6fe.json",
                "size_bytes": 2048,
            },
        ],
        "action": "list",
        "total_receipts": 6,
        "returned": 2,
        "receipt_directory":
            "/DO_NOT_RENDER/secret/receipt/path",
        "semantic_claims_verified": False,
        "production_authority": "OFF",
    }


def test_receipt_list_terminal_summary_is_exact_and_deterministic() -> None:
    result = _sample_receipt_list_result()

    expected = (
        "Bilingual Brain receipts: 6 total (showing 2)\n"
        "- 20261009T043350.153555Z-"
        "3d2b1e1fc81ad28d.json — 4096 bytes\n"
        "- 20261009T003543.155051Z-"
        "9ff0b4221c44f6fe.json — 2048 bytes\n"
        "\n"
        "Production authority: OFF. "
        "Semantic claims are not automatically verified."
    )

    first = (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
    )

    second = (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
    )

    assert first == expected
    assert second == expected
    assert first == second


def test_receipt_list_terminal_summary_excludes_untrusted_fields() -> None:
    summary = (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            _sample_receipt_list_result()
        )
    )

    assert summary

    forbidden = (
        "DO_NOT_RENDER_RECEIPT_CONTENT",
        "DO_NOT_RENDER_RECEIPT_PAYLOAD",
        "DO_NOT_RENDER_RECEIPT_DETAIL",
        "/DO_NOT_RENDER/secret/receipt/path",
        "receipt_directory",
    )

    for value in forbidden:
        assert value not in summary


@pytest.mark.parametrize(
    "action",
    [
        "latest",
        "read",
        "verify",
    ],
)
def test_receipt_terminal_summary_refuses_content_or_verify_actions(
    action: str,
) -> None:
    result = _sample_receipt_list_result()

    result["action"] = action

    # Simulate the receipt tool's latest/read behavior where results may
    # contain the parsed persisted receipt document.
    result["results"] = [
        {
            "arbitrary_workspace_receipt_content":
                "DO_NOT_TERMINALIZE_ME",
        },
    ]
    result["returned"] = 1

    assert (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
        == ""
    )


@pytest.mark.parametrize(
    "field, value",
    [
        (
            "semantic_claims_verified",
            True,
        ),
        (
            "production_authority",
            "ON",
        ),
    ],
)
def test_receipt_list_terminal_summary_requires_frozen_policy(
    field: str,
    value: object,
) -> None:
    result = _sample_receipt_list_result()
    result[field] = value

    assert (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
        == ""
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_filename",
        "bad_size",
        "bad_returned",
        "bad_results_type",
    ],
)
def test_receipt_list_terminal_summary_rejects_malformed_metadata(
    mutation: str,
) -> None:
    result = _sample_receipt_list_result()

    if mutation == "bad_filename":
        result["results"][0]["receipt"] = (
            "../untrusted.json"
        )
    elif mutation == "bad_size":
        result["results"][0]["size_bytes"] = True
    elif mutation == "bad_returned":
        result["returned"] = 1
    elif mutation == "bad_results_type":
        result["results"] = {
            "receipt": "not-a-list",
        }

    assert (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
        == ""
    )


def test_receipt_list_terminal_summary_handles_empty_list() -> None:
    result = _sample_receipt_list_result()

    result["results"] = []
    result["total_receipts"] = 0
    result["returned"] = 0

    summary = (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            result
        )
    )

    assert summary == (
        "Bilingual Brain receipts: 0 total (showing 0)\n"
        "- None\n"
        "\n"
        "Production authority: OFF. "
        "Semantic claims are not automatically verified."
    )


def test_receipt_list_terminal_path_breaks_before_next_model_round() -> None:
    source = inspect.getsource(
        agent_loop
    )

    render_marker = (
        'block.tool_type == "bilingual_brain_receipts"'
    )

    completed_marker = (
        "_receipt_list_terminal_completed = True"
    )

    break_marker = (
        "if _receipt_list_terminal_completed:"
    )

    next_round_marker = (
        "_append_tool_results("
        "messages, round_response, converted_calls,"
    )

    render_pos = source.index(
        render_marker
    )

    completed_pos = source.index(
        completed_marker,
        render_pos,
    )

    break_pos = source.index(
        break_marker,
        completed_pos,
    )

    next_round_pos = source.index(
        next_round_marker,
        break_pos,
    )

    assert (
        render_pos
        < completed_pos
        < break_pos
        < next_round_pos
    )

    assert (
        "safe deterministic metadata; "
        "skipping second model round"
        in source[
            break_pos:
            next_round_pos
        ]
    )


def test_receipt_v3_terminalization_is_explicitly_list_only() -> None:
    source = inspect.getsource(
        agent_loop
    )

    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary
    )

    assert (
        'result.get("action") != "list"'
        in helper
    )

    assert (
        "_receipt_list_terminal_completed"
        in source
    )

    # No independent content-reading terminal helper is allowed.
    forbidden_helpers = (
        "_receipt_read_terminal_summary",
        "_receipt_latest_terminal_summary",
        "_receipt_verify_terminal_summary",
    )

    for marker in forbidden_helpers:
        assert marker not in source
