from __future__ import annotations

import ast
from pathlib import Path

import src.agent_loop as agent_loop

from src.prompt_security import (
    untrusted_context_message,
)

from src.tool_capabilities import (
    messages_contain_external_untrusted_context,
)


EXACT_BRAIN_REQUEST = (
    "Analyze with the Bilingual Brain whether the current REVA "
    "evidence-receipt integration preserves a clean separation "
    "between evidence transport and semantic verification. "
    "Explain the structural result in English."
)


def domains_for(text: str) -> set[str]:
    result = agent_loop._classify_agent_request(
        [],
        text,
    )

    return set(
        result.get(
            "domains",
            set(),
        )
    )


def test_exact_brain_request_has_no_false_web_or_integration_domain():
    domains = domains_for(
        EXACT_BRAIN_REQUEST
    )

    assert "bilingual_brain" in domains
    assert "integrations" not in domains
    assert "web" not in domains


def test_real_integration_intent_still_selects_integrations():
    explicit_api = domains_for(
        "Use the api_call tool to call Home Assistant GET /api/states."
    )

    configured = domains_for(
        "Use my configured integration to query the service status."
    )

    assert "integrations" in explicit_api
    assert "integrations" in configured


def test_real_web_intent_still_selects_web():
    explicit_search = domains_for(
        "Search the web for current REVA release information."
    )

    weather = domains_for(
        "What is the current weather in Winnipeg?"
    )

    assert "web" in explicit_search
    assert "web" in weather


def test_agent_loop_marks_skill_wrapper_local_and_non_gate_arming():
    source = Path(
        agent_loop.__file__
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    matches = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if not (
            isinstance(
                func,
                ast.Name,
            )
            and func.id
            == "untrusted_context_message"
        ):
            continue

        if not node.args:
            continue

        first = node.args[0]

        if (
            isinstance(
                first,
                ast.Constant,
            )
            and first.value == "skills"
        ):
            matches.append(
                node
            )

    assert len(matches) == 1

    call = matches[0]

    kwargs = {
        kw.arg:
            kw.value
        for kw in call.keywords
        if kw.arg is not None
    }

    gate = kwargs.get(
        "arm_tool_gate"
    )

    assert isinstance(
        gate,
        ast.Constant,
    )

    assert gate.value is False

    origin = kwargs.get(
        "provenance_origin"
    )

    assert isinstance(
        origin,
        ast.Constant,
    )

    assert origin.value == "local"


def test_local_skill_data_remains_untrusted_but_does_not_arm_external_gate():
    message = (
        untrusted_context_message(
            "skills",
            "LOCAL USER-EDITABLE SKILL DATA",
            provenance_origin="local",
            arm_tool_gate=False,
        )
    )

    metadata = message[
        "metadata"
    ]

    assert (
        metadata["trusted"]
        is False
    )

    assert (
        metadata[
            "tool_gate_untrusted"
        ]
        is False
    )

    assert (
        metadata[
            "provenance_origin"
        ]
        == "local"
    )

    assert (
        messages_contain_external_untrusted_context(
            [message]
        )
        is False
    )


def test_real_external_context_still_arms_security_gate():
    message = (
        untrusted_context_message(
            "web search results",
            "EXTERNAL DATA",
            provenance_origin="external",
        )
    )

    metadata = message[
        "metadata"
    ]

    assert (
        metadata["trusted"]
        is False
    )

    assert (
        metadata[
            "tool_gate_untrusted"
        ]
        is True
    )

    assert (
        messages_contain_external_untrusted_context(
            [message]
        )
        is True
    )
