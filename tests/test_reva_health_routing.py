"""Regression tests for deterministic REVA-health routing."""

from __future__ import annotations

import inspect

import pytest

import src.agent_loop as agent_loop
from src.agent_loop import (
    _DOMAIN_RULES,
    _DOMAIN_TOOL_MAP,
    _bilingual_brain_receipt_args_from_text,
    _classify_agent_request,
    _reva_health_args_from_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "check REVA health",
        "show REVA health",
        "REVA health status",
        "is REVA healthy",
        "run REVA diagnostics",
        "diagnose REVA",
        "check the Bilingual Brain integration health",
    ],
)
def test_explicit_operational_reva_health_requests_route(
    text: str,
) -> None:
    assert (
        _reva_health_args_from_text(
            text
        )
        == {}
    )

    intent = _classify_agent_request(
        [],
        text,
    )

    domains = set(
        intent.get("domains")
        or set()
    )

    assert "reva_health" in domains
    assert "brain_receipts" not in domains
    assert "bilingual_brain" not in domains
    assert "web" not in domains
    assert "ui" not in domains
    assert intent["low_signal"] is False

    seeded_tools = set()

    for domain in domains:
        seeded_tools.update(
            _DOMAIN_TOOL_MAP.get(
                domain,
                set(),
            )
        )

    assert "reva_health" in seeded_tools


@pytest.mark.parametrize(
    "text",
    [
        "analyze REVA health architecture",
        "explain how REVA health works",
        "discuss the REVA health tool",
        "is quantum computing healthy for cryptography",
    ],
)
def test_health_discussion_or_unrelated_health_is_not_hijacked(
    text: str,
) -> None:
    assert (
        _reva_health_args_from_text(
            text
        )
        is None
    )

    intent = _classify_agent_request(
        [],
        text,
    )

    assert (
        "reva_health"
        not in set(
            intent.get("domains")
            or set()
        )
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "show my latest Bilingual Brain receipt",
            {
                "action": "latest",
            },
        ),
        (
            "verify my latest Bilingual Brain receipt",
            {
                "action": "latest",
            },
        ),
    ],
)
def test_receipt_requests_keep_receipt_routing_priority(
    text: str,
    expected: dict,
) -> None:
    assert (
        _reva_health_args_from_text(
            text
        )
        is None
    )

    assert (
        _bilingual_brain_receipt_args_from_text(
            text
        )
        == expected
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
    assert "reva_health" not in domains


def test_substantive_bilingual_brain_analysis_remains_brain_analysis() -> None:
    text = (
        "Analyze with the Bilingual Brain whether the current REVA "
        "evidence-receipt integration preserves a clean separation "
        "between evidence transport and semantic verification."
    )

    assert (
        _reva_health_args_from_text(
            text
        )
        is None
    )

    assert (
        _bilingual_brain_receipt_args_from_text(
            text
        )
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
    assert "reva_health" not in domains


def test_reva_health_domain_contract() -> None:
    assert (
        _DOMAIN_TOOL_MAP[
            "reva_health"
        ]
        == {
            "reva_health",
        }
    )

    rule = _DOMAIN_RULES[
        "reva_health"
    ]

    assert "read-only" in rule
    assert "PRODUCTION_AUTHORITY" in rule
    assert "receipt" in rule.lower()
    assert "Brain/model" in rule


def test_local_deepseek_deterministic_health_route_is_wired_before_receipt() -> None:
    source = inspect.getsource(
        agent_loop
    )

    health_predicate = (
        "_deterministic_reva_health_route = ("
    )
    health_branch = (
        "if _deterministic_reva_health_route:"
    )
    receipt_branch = (
        "elif _deterministic_receipt_route:"
    )

    assert health_predicate in source
    assert health_branch in source
    assert receipt_branch in source

    assert (
        source.index(
            health_branch
        )
        < source.index(
            receipt_branch
        )
    )

    assert (
        '"name": "reva_health"'
        in source
    )
    assert (
        '"deterministic_router": True'
        in source
    )
    assert (
        '"reva_health" in _relevant_tools'
        in source
    )
    assert (
        '"reva_health" in _tool_names_sent'
        in source
    )


def test_health_router_has_empty_argument_contract() -> None:
    assert (
        _reva_health_args_from_text(
            "check REVA health"
        )
        == {}
    )


def _sample_reva_health_result() -> dict:
    return {
        "service": "reva",
        "overall": "degraded",
        "results": [
            {
                "name": "tool_registry",
                "status": "ok",
                "detail":
                    "DO_NOT_RENDER_REGISTRY_DETAIL",
                "meta": {
                    "credential":
                        "DO_NOT_RENDER_REGISTRY_META",
                },
            },
            {
                "name": "brain_configuration",
                "status": "degraded",
                "detail":
                    "DO_NOT_RENDER_BRAIN_DETAIL",
                "meta": {
                    "password":
                        "DO_NOT_RENDER_PASSWORD",
                },
            },
            {
                "name": "receipt_configuration",
                "status": "disabled",
                "detail":
                    "DO_NOT_RENDER_RECEIPT_DIRECTORY",
                "meta": {
                    "path":
                        "/secret/receipt/path",
                },
            },
            {
                "name":
                    "latest_receipt_integrity",
                "status": "ok",
                "detail":
                    "DO_NOT_RENDER_RECEIPT_CONTENT",
                "meta": {
                    "receipt_payload":
                        "DO_NOT_RENDER_PAYLOAD",
                },
            },
            {
                "name":
                    "meshcore_policy_invariants",
                "status": "ok",
                "detail":
                    "DO_NOT_RENDER_POLICY_DETAIL",
                "meta": {
                    "private_key":
                        "DO_NOT_RENDER_PRIVATE_KEY",
                },
            },
        ],
        "network_access": False,
        "brain_executed": False,
        "model_executed": False,
        "receipt_generated": False,
        "semantic_claims_verified": False,
        "production_authority": "OFF",
        "secret_values_exposed": False,
    }


def test_reva_health_terminal_summary_is_exact_and_deterministic() -> None:
    result = _sample_reva_health_result()

    expected = (
        "REVA health: degraded\n"
        "- Tool registry: ok\n"
        "- Brain configuration: degraded\n"
        "- Receipt configuration: disabled\n"
        "- Latest receipt integrity: ok\n"
        "- MeshCore policy invariants: ok\n"
        "\n"
        "Production authority: OFF. "
        "Semantic claims are not automatically verified."
    )

    first = (
        agent_loop._reva_health_terminal_summary(
            result
        )
    )

    second = (
        agent_loop._reva_health_terminal_summary(
            result
        )
    )

    assert first == expected
    assert second == expected
    assert first == second


def test_reva_health_terminal_summary_does_not_render_detail_or_meta() -> None:
    summary = (
        agent_loop._reva_health_terminal_summary(
            _sample_reva_health_result()
        )
    )

    assert summary

    forbidden = (
        "DO_NOT_RENDER_REGISTRY_DETAIL",
        "DO_NOT_RENDER_REGISTRY_META",
        "DO_NOT_RENDER_BRAIN_DETAIL",
        "DO_NOT_RENDER_PASSWORD",
        "DO_NOT_RENDER_RECEIPT_DIRECTORY",
        "/secret/receipt/path",
        "DO_NOT_RENDER_RECEIPT_CONTENT",
        "DO_NOT_RENDER_PAYLOAD",
        "DO_NOT_RENDER_POLICY_DETAIL",
        "DO_NOT_RENDER_PRIVATE_KEY",
    )

    for value in forbidden:
        assert value not in summary


@pytest.mark.parametrize(
    "field, value",
    [
        ("network_access", True),
        ("brain_executed", True),
        ("model_executed", True),
        ("receipt_generated", True),
        (
            "semantic_claims_verified",
            True,
        ),
        (
            "production_authority",
            "ON",
        ),
        (
            "secret_values_exposed",
            True,
        ),
    ],
)
def test_reva_health_terminal_summary_refuses_broken_safety_contract(
    field: str,
    value: object,
) -> None:
    result = _sample_reva_health_result()
    result[field] = value

    assert (
        agent_loop._reva_health_terminal_summary(
            result
        )
        == ""
    )


def test_reva_health_terminal_summary_requires_complete_known_checks() -> None:
    result = _sample_reva_health_result()

    result["results"] = (
        result["results"][:-1]
    )

    assert (
        agent_loop._reva_health_terminal_summary(
            result
        )
        == ""
    )


def test_reva_health_terminal_path_breaks_before_next_model_round() -> None:
    source = inspect.getsource(
        agent_loop
    )

    render_marker = (
        'block.tool_type == "reva_health"'
    )

    completed_marker = (
        "_reva_health_terminal_completed = True"
    )

    break_marker = (
        "if _reva_health_terminal_completed:"
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
        "skipping second model round"
        in source[
            break_pos:
            next_round_pos
        ]
    )


def test_reva_health_terminal_path_preserves_existing_terminal_rules() -> None:
    source = inspect.getsource(
        agent_loop
    )

    assert (
        "if (_ody_notes_finetune_mode or "
        "_ody_qwen_finetune_model) "
        "and _ody_notes_tool_completed:"
        in source
    )

    assert (
        "if _ody_doc_tool_completed:"
        in source
    )

    assert (
        "if _doc_stream_create_completed:"
        in source
    )
