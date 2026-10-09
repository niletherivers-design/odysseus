import pytest

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
