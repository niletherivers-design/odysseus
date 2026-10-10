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
