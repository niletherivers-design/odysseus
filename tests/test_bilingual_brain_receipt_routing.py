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

    # V4 intentionally adds a strictly allowlisted verify-integrity
    # terminal helper.  Persisted receipt content readers must still
    # never receive deterministic terminalization.
    assert (
        "_bilingual_brain_receipt_verify_terminal_summary"
        in source
    )

    forbidden_content_helpers = (
        "_receipt_read_terminal_summary",
        "_receipt_latest_terminal_summary",
    )

    for marker in forbidden_content_helpers:
        assert marker not in source


def _sample_receipt_verify_result(
    *,
    verified: bool = True,
    filename_prefix: bool = True,
    expected_check: bool | None = True,
) -> dict:
    checks = {
        "filename_sha256_prefix":
            filename_prefix,
    }

    if expected_check is not None:
        checks["expected_sha256"] = (
            expected_check
        )

    return {
        "results": [
            {
                "verified":
                    verified,
                "sha256":
                    (
                        "3d2b1e1fc81ad28d"
                        "733b5be2915c4b7d"
                        "5077f27834467146"
                        "42c0222b45bbb94e"
                    ),
                "checks":
                    checks,
                "semantic_claims_verified":
                    False,
                "production_authority":
                    "OFF",

                # Deliberately hostile extras.  A deterministic terminal
                # projection must never render these.
                "content":
                    "DO_NOT_RENDER_CONTENT",
                "payload":
                    "DO_NOT_RENDER_PAYLOAD",
                "detail":
                    "DO_NOT_RENDER_DETAIL",
                "document": {
                    "message":
                        "DO_NOT_RENDER_DOCUMENT",
                },
            },
        ],
        "action":
            "verify",
        "receipt":
            (
                "20261009T043350.153555Z-"
                "3d2b1e1fc81ad28d.json"
            ),
        "semantic_claims_verified":
            False,
        "production_authority":
            "OFF",
        "receipt_directory":
            "/DO_NOT/RENDER/workspace/path",
    }


def test_receipt_verify_terminal_summary_exact_pass() -> None:
    result = _sample_receipt_verify_result()

    summary = (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            result
        )
    )

    expected = (
        "Bilingual Brain receipt integrity: PASS\n"
        "- Receipt: "
        "20261009T043350.153555Z-"
        "3d2b1e1fc81ad28d.json\n"
        "- SHA-256: "
        "3d2b1e1fc81ad28d"
        "733b5be2915c4b7d"
        "5077f27834467146"
        "42c0222b45bbb94e\n"
        "- Filename digest prefix: valid\n"
        "- Expected SHA-256: match\n"
        "\n"
        "Production authority: OFF. "
        "Receipt integrity does not automatically "
        "verify semantic claims."
    )

    assert summary == expected


def test_receipt_verify_terminal_summary_exact_fail() -> None:
    result = _sample_receipt_verify_result(
        verified=False,
        filename_prefix=False,
        expected_check=False,
    )

    summary = (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            result
        )
    )

    assert (
        "Bilingual Brain receipt integrity: FAIL"
        in summary
    )

    assert (
        "- Filename digest prefix: invalid"
        in summary
    )

    assert (
        "- Expected SHA-256: mismatch"
        in summary
    )


def test_receipt_verify_terminal_summary_allows_no_expected_digest() -> None:
    result = _sample_receipt_verify_result(
        expected_check=None,
    )

    summary = (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            result
        )
    )

    assert summary
    assert "- Expected SHA-256:" not in summary


def test_receipt_verify_terminal_summary_excludes_untrusted_content() -> None:
    summary = (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            _sample_receipt_verify_result()
        )
    )

    assert summary

    forbidden = (
        "DO_NOT_RENDER_CONTENT",
        "DO_NOT_RENDER_PAYLOAD",
        "DO_NOT_RENDER_DETAIL",
        "DO_NOT_RENDER_DOCUMENT",
        "/DO_NOT/RENDER/workspace/path",
        "receipt_directory",
        '"document"',
        '"payload"',
        '"content"',
        '"detail"',
    )

    for value in forbidden:
        assert value not in summary


@pytest.mark.parametrize(
    "action",
    [
        "list",
        "latest",
        "read",
    ],
)
def test_receipt_verify_terminal_summary_refuses_other_actions(
    action: str,
) -> None:
    result = _sample_receipt_verify_result()
    result["action"] = action

    assert (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            result
        )
        == ""
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_top_authority",
        "bad_top_semantic",
        "bad_receipt",
        "bad_results",
        "multiple_results",
        "bad_verification_authority",
        "bad_verification_semantic",
        "bad_verified_type",
        "bad_sha",
        "bad_checks",
        "bad_filename_check",
        "bad_expected_check",
    ],
)
def test_receipt_verify_terminal_summary_fails_closed(
    mutation: str,
) -> None:
    result = _sample_receipt_verify_result()

    verification = result["results"][0]

    if mutation == "bad_top_authority":
        result["production_authority"] = "ON"
    elif mutation == "bad_top_semantic":
        result["semantic_claims_verified"] = True
    elif mutation == "bad_receipt":
        result["receipt"] = "../receipt.json"
    elif mutation == "bad_results":
        result["results"] = {
            "verified": True,
        }
    elif mutation == "multiple_results":
        result["results"].append(
            dict(verification)
        )
    elif mutation == "bad_verification_authority":
        verification["production_authority"] = "ON"
    elif mutation == "bad_verification_semantic":
        verification["semantic_claims_verified"] = True
    elif mutation == "bad_verified_type":
        verification["verified"] = 1
    elif mutation == "bad_sha":
        verification["sha256"] = "xyz"
    elif mutation == "bad_checks":
        verification["checks"] = [
            "unsafe",
        ]
    elif mutation == "bad_filename_check":
        verification["checks"][
            "filename_sha256_prefix"
        ] = "yes"
    elif mutation == "bad_expected_check":
        verification["checks"][
            "expected_sha256"
        ] = "yes"

    assert (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            result
        )
        == ""
    )


def test_receipt_verify_terminal_source_is_verify_only() -> None:
    source = inspect.getsource(
        agent_loop
    )

    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary
    )

    assert (
        'result.get("action") != "verify"'
        in helper
    )

    assert (
        'result.get("production_authority") != "OFF"'
        in helper
    )

    assert (
        'result.get("semantic_claims_verified") is not False'
        in helper
    )

    assert (
        "_receipt_verify_terminal_completed"
        in source
    )

    assert (
        "_receipt_read_terminal_summary"
        not in source
    )

    assert (
        "_receipt_latest_terminal_summary"
        not in source
    )


def test_receipt_verify_terminal_break_precedes_next_model_round() -> None:
    source = inspect.getsource(
        agent_loop
    )

    render_marker = (
        "_receipt_verify_summary ="
    )

    complete_marker = (
        "_receipt_verify_terminal_completed = True"
    )

    break_marker = (
        "if _receipt_verify_terminal_completed:"
    )

    feed_marker = (
        "# Feed results back to LLM for next round"
    )

    render_pos = source.index(
        render_marker
    )

    complete_pos = source.index(
        complete_marker,
        render_pos,
    )

    break_pos = source.index(
        break_marker,
        complete_pos,
    )

    feed_pos = source.index(
        feed_marker,
        break_pos,
    )

    assert (
        render_pos
        < complete_pos
        < break_pos
        < feed_pos
    )

    assert (
        "safe deterministic integrity metadata"
        in source[
            break_pos:
            feed_pos
        ]
    )


def test_v3_list_terminalization_remains_independent() -> None:
    list_result = _sample_receipt_list_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            list_result
        )
    )

    verify_result = _sample_receipt_verify_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_list_terminal_summary(
            verify_result
        )
        == ""
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary(
            list_result
        )
        == ""
    )
