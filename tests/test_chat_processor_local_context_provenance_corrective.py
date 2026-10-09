from __future__ import annotations

import ast
from pathlib import Path

from src.prompt_security import (
    untrusted_context_message,
)

from src.tool_capabilities import (
    messages_contain_external_untrusted_context,
)


ROOT = Path("/app")

TARGET = (
    ROOT
    / "src"
    / "chat_processor.py"
)

WANTED = {
    "saved memory: pinned context",
    "saved memory: retrieved context",
    "available skills index",
}


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def _calls():
    tree = ast.parse(
        TARGET.read_text(
            encoding="utf-8",
        ),
        filename=str(TARGET),
    )

    found = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if (
            _call_name(node.func)
            != "untrusted_context_message"
        ):
            continue

        if not node.args:
            continue

        first = node.args[0]

        if not (
            isinstance(
                first,
                ast.Constant,
            )
            and isinstance(
                first.value,
                str,
            )
        ):
            continue

        if first.value not in WANTED:
            continue

        found.append(
            (
                first.value,
                node,
            )
        )

    return found


def test_exact_three_chat_preface_local_context_calls():
    calls = _calls()

    assert len(calls) == 3

    assert {
        label
        for label, _
        in calls
    } == WANTED


def test_three_calls_are_local_but_remain_untrusted():
    for label, node in _calls():
        keywords = {
            kw.arg:
                kw.value
            for kw in node.keywords
            if kw.arg
        }

        provenance = keywords[
            "provenance_origin"
        ]

        gate = keywords[
            "arm_tool_gate"
        ]

        assert isinstance(
            provenance,
            ast.Constant,
        ), label

        assert (
            provenance.value
            == "local"
        ), label

        assert isinstance(
            gate,
            ast.Constant,
        ), label

        assert (
            gate.value
            is False
        ), label


def test_local_saved_memory_does_not_arm_external_gate():
    for label in (
        "saved memory: pinned context",
        "saved memory: retrieved context",
    ):
        msg = (
            untrusted_context_message(
                label,
                "LOCAL SAVED MEMORY",
                provenance_origin="local",
                arm_tool_gate=False,
            )
        )

        md = msg["metadata"]

        assert md["trusted"] is False
        assert (
            md["tool_gate_untrusted"]
            is False
        )

        assert (
            md["provenance_origin"]
            == "local"
        )

        assert (
            messages_contain_external_untrusted_context(
                [msg]
            )
            is False
        )


def test_local_skills_index_does_not_arm_external_gate():
    msg = (
        untrusted_context_message(
            "available skills index",
            "LOCAL USER-EDITABLE INDEX",
            provenance_origin="local",
            arm_tool_gate=False,
        )
    )

    md = msg["metadata"]

    assert md["trusted"] is False

    assert (
        md["tool_gate_untrusted"]
        is False
    )

    assert (
        md["provenance_origin"]
        == "local"
    )

    assert (
        messages_contain_external_untrusted_context(
            [msg]
        )
        is False
    )


def test_genuine_external_context_still_arms_gate():
    msg = (
        untrusted_context_message(
            "web search results",
            "EXTERNAL DATA",
            provenance_origin="external",
        )
    )

    assert (
        messages_contain_external_untrusted_context(
            [msg]
        )
        is True
    )
