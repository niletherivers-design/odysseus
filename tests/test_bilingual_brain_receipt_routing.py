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
    )

    for marker in forbidden_content_helpers:
        assert marker not in source

    assert (
        "_bilingual_brain_receipt_latest_terminal_summary"
        in source
    )


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


def test_receipt_verify_terminal_source_remains_bounded() -> None:
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


# ---------------------------------------------------------------------------
# REVA Orchestration v5 — deterministic expected-SHA256 receipt routing
# ---------------------------------------------------------------------------

_V5_RECEIPT = (
    "20261009T043350.153555Z-"
    "3d2b1e1fc81ad28d.json"
)

_V5_EXPECTED_SHA = (
    "3d2b1e1fc81ad28d"
    "733b5be2915c4b7d"
    "5077f27834467146"
    "42c0222b45bbb94e"
)


def test_v5_receipt_router_supplies_expected_sha256() -> None:
    text = (
        "verify Bilingual Brain receipt "
        f"{_V5_RECEIPT} "
        "against SHA-256 "
        f"{_V5_EXPECTED_SHA}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        == {
            "action":
                "verify",
            "receipt":
                _V5_RECEIPT,
            "expected_sha256":
                _V5_EXPECTED_SHA,
        }
    )


def test_v5_receipt_router_normalizes_expected_sha256() -> None:
    text = (
        "validate Bilingual Brain receipt "
        f"{_V5_RECEIPT} "
        "against SHA256: "
        f"{_V5_EXPECTED_SHA.upper()}"
    )

    result = (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
    )

    assert result == {
        "action":
            "verify",
        "receipt":
            _V5_RECEIPT,
        "expected_sha256":
            _V5_EXPECTED_SHA,
    }


def test_v5_receipt_router_preserves_plain_verify() -> None:
    text = (
        "verify Bilingual Brain receipt "
        f"{_V5_RECEIPT}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        == {
            "action":
                "verify",
            "receipt":
                _V5_RECEIPT,
        }
    )


@pytest.mark.parametrize(
    "digest",
    (
        "a" * 63,
        "a" * 65,
        "not-a-sha256-digest",
    ),
)
def test_v5_receipt_router_rejects_malformed_expected_sha256(
    digest: str,
) -> None:
    text = (
        "verify Bilingual Brain receipt "
        f"{_V5_RECEIPT} "
        "against SHA-256 "
        f"{digest}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


def test_v5_receipt_router_rejects_multiple_sha256_candidates() -> None:
    text = (
        "verify Bilingual Brain receipt "
        f"{_V5_RECEIPT} "
        "against SHA-256 "
        f"{'a' * 64} "
        "and SHA-256 "
        f"{'b' * 64}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


def test_v5_expected_sha256_requires_named_receipt() -> None:
    text = (
        "verify my latest Bilingual Brain receipt "
        "against SHA-256 "
        f"{_V5_EXPECTED_SHA}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


def test_v5_expected_sha256_preserves_no_generic_read_terminal_helper() -> None:
    source = inspect.getsource(
        agent_loop
    )

    assert (
        "_receipt_read_terminal_summary"
        not in source
    )

    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_verify_terminal_summary
    )

    assert (
        '"expected_sha256"'
        in helper
    )

    assert (
        "- Expected SHA-256: "
        in helper
    )


def test_v5_router_expected_sha256_is_bounded_projection_only() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_args_from_text
    )

    assert (
        "expected_sha256_matches"
        in helper
    )

    assert (
        r"([0-9a-fA-F]{64})"
        in helper
    )

    assert (
        "len(expected_sha256_matches) > 1"
        in helper
    )

    assert (
        'args["expected_sha256"] = expected_sha256'
        in helper
    )


def test_v5_receipt_router_rejects_valid_plus_malformed_sha256_candidate() -> None:
    text = (
        "verify Bilingual Brain receipt "
        f"{_V5_RECEIPT} "
        "against SHA-256 "
        f"{_V5_EXPECTED_SHA} "
        "and SHA-256 not-a-valid-digest"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


def test_v5_router_counts_all_sha256_labels_before_selection() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_args_from_text
    )

    assert (
        "expected_sha256_labels"
        in helper
    )

    assert (
        "len(expected_sha256_labels) > 1"
        in helper
    )


# ---------------------------------------------------------------------------
# REVA Orchestration v6 — deterministic latest-receipt metadata projection
# ---------------------------------------------------------------------------

_V6_LATEST_RECEIPT = (
    "20261009T043350.153555Z-"
    "3d2b1e1fc81ad28d.json"
)

_V6_LATEST_SHA = (
    "3d2b1e1fc81ad28d"
    "733b5be2915c4b7d"
    "5077f27834467146"
    "42c0222b45bbb94e"
)


def _sample_v6_latest_result(
    *,
    filename_prefix_valid: bool = True,
    receipt_policy_valid: bool = True,
) -> dict:
    return {
        "action":
            "latest",
        "receipt":
            _V6_LATEST_RECEIPT,
        "receipt_sha256":
            _V6_LATEST_SHA,
        "filename_sha256_prefix_valid":
            filename_prefix_valid,
        "receipt_policy_valid":
            receipt_policy_valid,
        "semantic_claims_verified":
            False,
        "production_authority":
            "OFF",
        "results": [
            {
                "created_at":
                    "DO_NOT_RENDER_CREATED_AT",
                "schema":
                    "DO_NOT_RENDER_SCHEMA",
                "policy": {
                    "secret":
                        "DO_NOT_RENDER_POLICY",
                },
                "request": {
                    "text":
                        "DO_NOT_RENDER_REQUEST",
                },
                "response": {
                    "text":
                        "DO_NOT_RENDER_RESPONSE",
                },
                "content":
                    "DO_NOT_RENDER_CONTENT",
                "payload":
                    "DO_NOT_RENDER_PAYLOAD",
                "detail":
                    "DO_NOT_RENDER_DETAIL",
                "document": {
                    "text":
                        "DO_NOT_RENDER_DOCUMENT",
                },
            },
        ],
        "receipt_directory":
            "/DO_NOT/RENDER/workspace/path",
    }


def test_v6_latest_terminal_summary_is_exact_and_deterministic() -> None:
    result = _sample_v6_latest_result()

    summary = (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            result
        )
    )

    assert summary == (
        "Latest Bilingual Brain receipt metadata\n"
        f"- Receipt: {_V6_LATEST_RECEIPT}\n"
        f"- Reported SHA-256: {_V6_LATEST_SHA}\n"
        "- Filename digest prefix: valid\n"
        "- Receipt policy: valid\n"
        "\n"
        "Production authority: OFF. "
        "Receipt metadata is workspace-untrusted and does not "
        "automatically verify semantic claims."
    )


def test_v6_latest_terminal_summary_projects_invalid_flags_only() -> None:
    result = _sample_v6_latest_result(
        filename_prefix_valid=False,
        receipt_policy_valid=False,
    )

    summary = (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            result
        )
    )

    assert (
        "- Filename digest prefix: invalid"
        in summary
    )

    assert (
        "- Receipt policy: invalid"
        in summary
    )


def test_v6_latest_terminal_summary_never_renders_untrusted_result_content() -> None:
    summary = (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            _sample_v6_latest_result()
        )
    )

    forbidden = (
        "DO_NOT_RENDER_CREATED_AT",
        "DO_NOT_RENDER_SCHEMA",
        "DO_NOT_RENDER_POLICY",
        "DO_NOT_RENDER_REQUEST",
        "DO_NOT_RENDER_RESPONSE",
        "DO_NOT_RENDER_CONTENT",
        "DO_NOT_RENDER_PAYLOAD",
        "DO_NOT_RENDER_DETAIL",
        "DO_NOT_RENDER_DOCUMENT",
        "/DO_NOT/RENDER/workspace/path",
        "receipt_directory",
        '"request"',
        '"response"',
        '"payload"',
        '"document"',
    )

    for marker in forbidden:
        assert marker not in summary


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("action", "read"),
        ("production_authority", "ON"),
        ("semantic_claims_verified", True),
        ("receipt", "../unsafe.json"),
        ("receipt_sha256", "a" * 63),
        ("receipt_sha256", "A" * 64),
        (
            "filename_sha256_prefix_valid",
            1,
        ),
        (
            "receipt_policy_valid",
            "true",
        ),
        ("results", []),
        (
            "results",
            [{}, {}],
        ),
        (
            "results",
            ["not-a-dict"],
        ),
    ),
)
def test_v6_latest_terminal_summary_fails_closed(
    field,
    value,
) -> None:
    result = _sample_v6_latest_result()
    result[field] = value

    assert (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            result
        )
        == ""
    )


def test_v6_latest_terminal_summary_requires_false_semantic_claim() -> None:
    result = _sample_v6_latest_result()

    result.pop(
        "semantic_claims_verified"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            result
        )
        == ""
    )


def test_v6_latest_router_uses_existing_latest_action() -> None:
    args = (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            "show me my latest Bilingual Brain receipt"
        )
    )

    assert args == {
        "action":
            "latest",
    }


def test_v6_latest_helper_is_latest_only() -> None:
    latest_result = _sample_v6_latest_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            latest_result
        )
    )

    list_result = _sample_receipt_list_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            list_result
        )
        == ""
    )

    verify_result = _sample_receipt_verify_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary(
            verify_result
        )
        == ""
    )


def test_v6_read_generic_unbounded_terminalization_remains_absent() -> None:
    source = inspect.getsource(
        agent_loop
    )

    assert (
        "_receipt_read_terminal_summary"
        not in source
    )

    assert (
        "_receipt_read_terminal_completed"
        not in source
    )


def test_v6_latest_terminal_path_breaks_before_next_model_round() -> None:
    source = inspect.getsource(
        agent_loop
    )

    render_marker = (
        "_receipt_latest_summary ="
    )

    completed_marker = (
        "_receipt_latest_terminal_completed = True"
    )

    break_marker = (
        "if _receipt_latest_terminal_completed:"
    )

    feed_marker = (
        "# Feed results back to LLM for next round"
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

    feed_pos = source.index(
        feed_marker,
        break_pos,
    )

    assert (
        render_pos
        < completed_pos
        < break_pos
        < feed_pos
    )

    assert (
        "safe deterministic metadata; "
        in source[
            break_pos:
            feed_pos
        ]
    )

    assert (
        "skipping second model round"
        in source[
            break_pos:
            feed_pos
        ]
    )


def test_v6_latest_helper_uses_only_bounded_top_level_metadata() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_latest_terminal_summary
    )

    for required in (
        'result.get("action") != "latest"',
        'result.get("production_authority") != "OFF"',
        'result.get("semantic_claims_verified") is not False',
        'result.get("receipt")',
        '"receipt_sha256"',
        '"filename_sha256_prefix_valid"',
        '"receipt_policy_valid"',
        r"[0-9a-f]{64}",
    ):
        assert required in helper

    for forbidden in (
        'results[0].get("request")',
        'results[0].get("response")',
        'results[0].get("schema")',
        'results[0].get("policy")',
        'results[0].get("content")',
        'results[0].get("payload")',
        'results[0].get("detail")',
        'results[0].get("document")',
        'result.get("receipt_directory")',
    ):
        assert forbidden not in helper


def test_v6_list_and_verify_terminalization_remain_preserved() -> None:
    source = inspect.getsource(
        agent_loop
    )

    assert (
        "_bilingual_brain_receipt_list_terminal_summary"
        in source
    )

    assert (
        "_bilingual_brain_receipt_verify_terminal_summary"
        in source
    )

    assert (
        "_receipt_list_terminal_completed = True"
        in source
    )

    assert (
        "_receipt_verify_terminal_completed = True"
        in source
    )

    assert (
        "_receipt_latest_terminal_completed = True"
        in source
    )


# ---------------------------------------------------------------------------
# REVA Orchestration v7 — strict deterministic receipt-read allowlist
# ---------------------------------------------------------------------------

_V7_READ_RECEIPT = (
    "20261009T043350.153555Z-"
    "3d2b1e1fc81ad28d.json"
)

_V7_READ_SHA = (
    "3d2b1e1fc81ad28d"
    "733b5be2915c4b7d"
    "5077f27834467146"
    "42c0222b45bbb94e"
)


def _sample_v7_read_result() -> dict:
    return {
        "action":
            "read",
        "receipt":
            _V7_READ_RECEIPT,
        "receipt_sha256":
            _V7_READ_SHA,
        "filename_sha256_prefix_valid":
            True,
        "receipt_policy_valid":
            True,
        "semantic_claims_verified":
            False,
        "production_authority":
            "OFF",
        "results": [
            {
                "created_at":
                    "DO_NOT_RENDER_CREATED_AT",
                "schema":
                    "DO_NOT_RENDER_SCHEMA",
                "policy": {
                    "secret":
                        "DO_NOT_RENDER_POLICY",
                },
                "request": {
                    "text":
                        "DO_NOT_RENDER_REQUEST",
                },
                "response": {
                    "text":
                        "DO_NOT_RENDER_RESPONSE",
                },
                "content":
                    "DO_NOT_RENDER_CONTENT",
                "payload":
                    "DO_NOT_RENDER_PAYLOAD",
                "detail":
                    "DO_NOT_RENDER_DETAIL",
                "document": {
                    "text":
                        "DO_NOT_RENDER_DOCUMENT",
                },
            },
        ],
        "receipt_directory":
            "/DO_NOT/RENDER/workspace/path",
    }


def test_v7_read_allowlist_summary_is_exact_and_deterministic() -> None:
    result = _sample_v7_read_result()

    summary = (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            result
        )
    )

    assert summary == (
        "Bilingual Brain receipt read projection\n"
        f"- Receipt: {_V7_READ_RECEIPT}\n"
        f"- Reported SHA-256: {_V7_READ_SHA}\n"
        "- Filename digest prefix: valid\n"
        "- Receipt policy: valid\n"
        "- Receipt body: withheld\n"
        "\n"
        "Production authority: OFF. "
        "Receipt body is workspace-untrusted and is not rendered. "
        "Receipt metadata does not automatically verify semantic claims."
    )


def test_v7_read_allowlist_never_renders_receipt_body_values() -> None:
    result = _sample_v7_read_result()

    summary = (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            result
        )
    )

    forbidden = (
        "DO_NOT_RENDER_CREATED_AT",
        "DO_NOT_RENDER_SCHEMA",
        "DO_NOT_RENDER_POLICY",
        "DO_NOT_RENDER_REQUEST",
        "DO_NOT_RENDER_RESPONSE",
        "DO_NOT_RENDER_CONTENT",
        "DO_NOT_RENDER_PAYLOAD",
        "DO_NOT_RENDER_DETAIL",
        "DO_NOT_RENDER_DOCUMENT",
        "/DO_NOT/RENDER/workspace/path",
    )

    for marker in forbidden:
        assert marker not in summary


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("action", "latest"),
        ("error", "read failed"),
        ("production_authority", "ON"),
        ("semantic_claims_verified", True),
        ("receipt", "../unsafe.json"),
        ("receipt_sha256", "a" * 63),
        ("receipt_sha256", "A" * 64),
        (
            "receipt_sha256",
            "0" * 64,
        ),
        (
            "filename_sha256_prefix_valid",
            False,
        ),
        (
            "filename_sha256_prefix_valid",
            1,
        ),
        (
            "receipt_policy_valid",
            False,
        ),
        (
            "receipt_policy_valid",
            "true",
        ),
        ("results", []),
        (
            "results",
            [{}, {}],
        ),
        (
            "results",
            ["not-a-dict"],
        ),
        (
            "results",
            [{}],
        ),
    ),
)
def test_v7_read_allowlist_fails_closed(
    field,
    value,
) -> None:
    result = _sample_v7_read_result()
    result[field] = value

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            result
        )
        == ""
    )


def test_v7_read_allowlist_requires_false_semantic_claim() -> None:
    result = _sample_v7_read_result()

    result.pop(
        "semantic_claims_verified"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            result
        )
        == ""
    )


def test_v7_read_router_uses_existing_named_receipt_action() -> None:
    text = (
        "read Brain receipt "
        f"{_V7_READ_RECEIPT}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        == {
            "action":
                "read",
            "receipt":
                _V7_READ_RECEIPT,
        }
    )


def test_v7_read_allowlist_helper_is_read_only() -> None:
    read_result = _sample_v7_read_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            read_result
        )
    )

    latest_result = _sample_v6_latest_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            latest_result
        )
        == ""
    )

    list_result = _sample_receipt_list_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            list_result
        )
        == ""
    )

    verify_result = _sample_receipt_verify_result()

    assert (
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary(
            verify_result
        )
        == ""
    )


def test_v7_read_allowlist_uses_no_untrusted_body_values() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary
    )

    for required in (
        'result.get("action") != "read"',
        'result.get("production_authority") != "OFF"',
        'result.get("semantic_claims_verified") is not False',
        'result.get("receipt")',
        '"receipt_sha256"',
        '"filename_sha256_prefix_valid"',
        '"receipt_policy_valid"',
        '"created_at"',
        '"schema"',
        '"policy"',
        '"request"',
        '"response"',
        r"[0-9a-f]{64}",
    ):
        assert required in helper

    for forbidden in (
        'item.get("created_at")',
        'item.get("schema")',
        'item.get("policy")',
        'item.get("request")',
        'item.get("response")',
        'item.get("content")',
        'item.get("payload")',
        'item.get("detail")',
        'item.get("document")',
        'results[0].get("created_at")',
        'results[0].get("schema")',
        'results[0].get("policy")',
        'results[0].get("request")',
        'results[0].get("response")',
        'result.get("receipt_directory")',
    ):
        assert forbidden not in helper


def test_v7_generic_unbounded_read_terminal_helper_remains_absent() -> None:
    source = inspect.getsource(
        agent_loop
    )

    assert (
        "_receipt_read_terminal_summary"
        not in source
    )

    assert (
        "_receipt_read_terminal_completed"
        not in source
    )

    assert (
        "_bilingual_brain_receipt_read_allowlist_summary"
        in source
    )

    assert (
        "_receipt_read_allowlist_completed"
        in source
    )


def test_v7_read_allowlist_breaks_before_next_model_round() -> None:
    source = inspect.getsource(
        agent_loop
    )

    render_marker = (
        "_receipt_read_projection ="
    )

    completed_marker = (
        "_receipt_read_allowlist_completed = True"
    )

    break_marker = (
        "if _receipt_read_allowlist_completed:"
    )

    feed_marker = (
        "# Feed results back to LLM for next round"
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

    feed_pos = source.index(
        feed_marker,
        break_pos,
    )

    assert (
        render_pos
        < completed_pos
        < break_pos
        < feed_pos
    )

    assert (
        "strict safe allowlist metadata"
        in source[
            break_pos:
            feed_pos
        ]
    )

    assert (
        "skipping second model round"
        in source[
            break_pos:
            feed_pos
        ]
    )


def test_v7_predecessor_terminalizations_remain_preserved() -> None:
    source = inspect.getsource(
        agent_loop
    )

    for helper in (
        "_bilingual_brain_receipt_list_terminal_summary",
        "_bilingual_brain_receipt_verify_terminal_summary",
        "_bilingual_brain_receipt_latest_terminal_summary",
        "_bilingual_brain_receipt_read_allowlist_summary",
    ):
        assert helper in source

    for completed in (
        "_receipt_list_terminal_completed = True",
        "_receipt_verify_terminal_completed = True",
        "_receipt_latest_terminal_completed = True",
        "_receipt_read_allowlist_completed = True",
    ):
        assert completed in source


def test_v7_read_allowlist_does_not_upgrade_result_integrity() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_read_allowlist_summary
    )

    assert "WORKSPACE_TRUSTED" not in helper
    assert "TRUSTED" not in helper

    assert (
        "workspace-untrusted"
        in helper
    )

    assert (
        "does not automatically verify semantic claims"
        in helper
    )


# ---------------------------------------------------------------------------
# REVA Orchestration v8 — compound receipt intents fail closed
# ---------------------------------------------------------------------------

_V8_RECEIPT = (
    "20261009T043350.153555Z-"
    "3d2b1e1fc81ad28d.json"
)

_V8_SHA256 = (
    "3d2b1e1fc81ad28d"
    "733b5be2915c4b7d"
    "5077f27834467146"
    "42c0222b45bbb94e"
)


@pytest.mark.parametrize(
    "text",
    [
        "list and show me my latest Bilingual Brain receipt",
        (
            "show my latest receipt and verify receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "list receipts and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "and list all receipts"
        ),
        (
            "show latest Brain receipt and read "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify Brain receipt "
            f"{_V8_RECEIPT} "
            "expected sha256 "
            f"{_V8_SHA256} "
            "and read it"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "then verify it"
        ),
        (
            "latest and list and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
    ],
)
def test_v8_compound_receipt_intents_fail_closed(
    text: str,
) -> None:
    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "list and show me my latest Bilingual Brain receipt",
        (
            "show my latest receipt and verify receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "list receipts and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "and list all receipts"
        ),
        (
            "show latest Brain receipt and read "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify Brain receipt "
            f"{_V8_RECEIPT} "
            "expected sha256 "
            f"{_V8_SHA256} "
            "and read it"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "then verify it"
        ),
        (
            "latest and list and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
    ],
)
def test_v8_ambiguous_receipt_intents_do_not_escape_to_web_or_ui(
    text: str,
) -> None:
    intent = (
        agent_loop
        ._classify_agent_request(
            [],
            text,
        )
    )

    domains = set(
        intent.get("domains")
        or set()
    )

    assert "brain_receipts" not in domains
    assert "web" not in domains
    assert "ui" not in domains


def test_v8_single_receipt_intents_remain_preserved() -> None:
    cases = [
        (
            "list all my Bilingual Brain receipts",
            {
                "action":
                    "list",
                "limit":
                    10,
            },
        ),
        (
            "show me my latest Bilingual Brain receipt",
            {
                "action":
                    "latest",
            },
        ),
        (
            f"read Brain receipt {_V8_RECEIPT}",
            {
                "action":
                    "read",
                "receipt":
                    _V8_RECEIPT,
            },
        ),
        (
            f"verify Brain receipt {_V8_RECEIPT}",
            {
                "action":
                    "verify",
                "receipt":
                    _V8_RECEIPT,
            },
        ),
        (
            (
                "verify Brain receipt "
                f"{_V8_RECEIPT} "
                "expected sha256 "
                f"{_V8_SHA256}"
            ),
            {
                "action":
                    "verify",
                "receipt":
                    _V8_RECEIPT,
                "expected_sha256":
                    _V8_SHA256,
            },
        ),
    ]

    for text, expected in cases:
        assert (
            agent_loop
            ._bilingual_brain_receipt_args_from_text(
                text
            )
            == expected
        )


@pytest.mark.parametrize(
    "text",
    [
        "verify my latest Bilingual Brain receipt",
        "read my latest Bilingual Brain receipt",
    ],
)
def test_v8_unnamed_latest_convenience_workflow_is_preserved(
    text: str,
) -> None:
    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        == {
            "action":
                "latest",
        }
    )


def test_v8_ambiguity_guard_is_receipt_context_bounded() -> None:
    assert (
        agent_loop
        ._bilingual_brain_receipt_request_is_ambiguous(
            "read and verify the API documentation"
        )
        is False
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_explicit_actions(
            "read and verify the API documentation"
        )
        == set()
    )


def test_v8_ambiguity_guard_has_no_receipt_or_tool_side_effects() -> None:
    helper = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_request_is_ambiguous
    )

    classifier = inspect.getsource(
        agent_loop
        ._classify_agent_request
    )

    assert "bilingual_brain_receipts(" not in helper
    assert "open(" not in helper
    assert "Path(" not in helper
    assert "subprocess" not in helper

    assert "_receipt_ambiguous" in classifier
    assert "not _receipt_ambiguous" in classifier


def test_v8_ambiguity_guard_runs_before_receipt_precedence() -> None:
    router = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_args_from_text
    )

    guard = router.index(
        "_bilingual_brain_receipt_request_is_ambiguous"
    )

    named = router.index(
        "if receipt_name is not None:"
    )

    listing = router.index(
        're.search(r"\\breceipts\\b", q)'
    )

    assert guard < named
    assert guard < listing


def test_v8_predecessor_terminalization_helpers_remain_preserved() -> None:
    source = inspect.getsource(
        agent_loop
    )

    for helper in (
        "_bilingual_brain_receipt_list_terminal_summary",
        "_bilingual_brain_receipt_verify_terminal_summary",
        "_bilingual_brain_receipt_latest_terminal_summary",
        "_bilingual_brain_receipt_read_allowlist_summary",
    ):
        assert helper in source

    assert (
        "_receipt_read_terminal_summary"
        not in source
    )

    assert (
        "_receipt_read_terminal_completed"
        not in source
    )


def test_v8_ambiguous_receipt_classifier_returns_zero_domains() -> None:
    examples = [
        "list and show me my latest Bilingual Brain receipt",
        (
            "read and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "list receipts and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "show latest Brain receipt and read "
            f"{_V8_RECEIPT}"
        ),
    ]

    for text in examples:
        intent = (
            agent_loop
            ._classify_agent_request(
                [],
                text,
            )
        )

        assert intent["domains"] == set()
        assert intent["low_signal"] is True


def test_v8_canonical_filename_establishes_receipt_context_for_ambiguity() -> None:
    text = (
        "show my latest receipt and verify receipt "
        f"{_V8_RECEIPT}"
    )

    actions = (
        agent_loop
        ._bilingual_brain_receipt_explicit_actions(
            text
        )
    )

    assert actions == {
        "latest",
        "verify",
    }

    assert (
        agent_loop
        ._bilingual_brain_receipt_request_is_ambiguous(
            text
        )
        is True
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )

    intent = (
        agent_loop
        ._classify_agent_request(
            [],
            text,
        )
    )

    assert intent["domains"] == set()
    assert intent["low_signal"] is True


def test_v8_canonical_filename_context_does_not_hijack_ordinary_receipts() -> None:
    text = (
        "show my latest grocery receipt and verify the total"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_explicit_actions(
            text
        )
        == set()
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_request_is_ambiguous(
            text
        )
        is False
    )


# ---------------------------------------------------------------------------
# REVA Orchestration v9 — deterministic ambiguous receipt clarification
# ---------------------------------------------------------------------------


def test_v9_ambiguous_receipt_terminal_summary_is_bounded() -> None:
    text = (
        "read and verify Brain receipt "
        f"{_V8_RECEIPT}"
    )

    summary = (
        agent_loop
        ._bilingual_brain_receipt_ambiguity_terminal_summary(
            text
        )
    )

    assert summary == (
        "Bilingual Brain receipt request is ambiguous\n"
        "- Requested actions: read, verify\n"
        "- Receipt operation: not run\n"
        "\n"
        "Choose exactly one action: list receipts, show the latest "
        "receipt, read one named receipt, or verify one named receipt "
        "(optionally with an expected SHA-256).\n"
        "\n"
        "Production authority: OFF. No receipt tool was executed, "
        "and receipt contents were not read or rendered."
    )

    assert _V8_RECEIPT not in summary
    assert _V8_SHA256 not in summary
    assert "/app/workspace" not in summary


@pytest.mark.parametrize(
    "text",
    [
        "list and show me my latest Bilingual Brain receipt",
        (
            "show my latest receipt and verify receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "list receipts and read Brain receipt "
            f"{_V8_RECEIPT}"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "and list all receipts"
        ),
        (
            "show latest Brain receipt and read "
            f"{_V8_RECEIPT}"
        ),
        (
            "verify Brain receipt "
            f"{_V8_RECEIPT} "
            "expected sha256 "
            f"{_V8_SHA256} "
            "and read it"
        ),
        (
            "read Brain receipt "
            f"{_V8_RECEIPT} "
            "then verify it"
        ),
        (
            "latest and list and verify Brain receipt "
            f"{_V8_RECEIPT}"
        ),
    ],
)
def test_v9_all_v8_ambiguity_cases_get_clarification(
    text: str,
) -> None:
    summary = (
        agent_loop
        ._bilingual_brain_receipt_ambiguity_terminal_summary(
            text
        )
    )

    assert summary
    assert "request is ambiguous" in summary
    assert "Receipt operation: not run" in summary
    assert "Production authority: OFF." in summary

    assert _V8_RECEIPT not in summary
    assert _V8_SHA256 not in summary


@pytest.mark.parametrize(
    "text",
    [
        "list all my Bilingual Brain receipts",
        "show me my latest Bilingual Brain receipt",
        f"read Brain receipt {_V8_RECEIPT}",
        f"verify Brain receipt {_V8_RECEIPT}",
        "read and verify the API documentation",
        "show my latest grocery receipt and verify the total",
    ],
)
def test_v9_nonambiguous_requests_do_not_get_receipt_clarification(
    text: str,
) -> None:
    assert (
        agent_loop
        ._bilingual_brain_receipt_ambiguity_terminal_summary(
            text
        )
        == ""
    )


def test_v9_clarification_helper_has_no_io_or_tool_execution() -> None:
    source = inspect.getsource(
        agent_loop
        ._bilingual_brain_receipt_ambiguity_terminal_summary
    )

    for forbidden in (
        "open(",
        "Path(",
        "subprocess",
        "bilingual_brain_receipts(",
        "execute_tool",
        "llm_call",
    ):
        assert forbidden not in source


def test_v9_clarification_runs_before_model_and_tool_pipeline() -> None:
    source = inspect.getsource(
        agent_loop.stream_agent_loop
    )

    classifier = source.index(
        "_intent = _classify_agent_request"
    )

    clarification = source.index(
        "_receipt_ambiguity_summary ="
    )

    terminal = source.index(
        "if _receipt_ambiguity_summary:"
    )

    low_signal = source.index(
        "_low_signal_turn ="
    )

    tool_seed = source.index(
        "If deterministic domain detection fired"
    )

    round_loop = source.index(
        "for round_num in range"
    )

    assert classifier < clarification
    assert clarification < terminal
    assert terminal < low_signal
    assert terminal < tool_seed
    assert terminal < round_loop


def test_v9_clarification_terminal_path_is_zero_round_zero_tool() -> None:
    source = inspect.getsource(
        agent_loop.stream_agent_loop
    )

    start = source.index(
        "if _receipt_ambiguity_summary:"
    )

    end = source.index(
        "_low_signal_turn =",
        start,
    )

    block = source[
        start:end
    ]

    assert '"agent_rounds":' in block
    assert '"tool_calls":' in block
    assert '"ambiguous_receipt":' in block
    assert '"deterministic_terminal":' in block

    assert "agent_rounds" in block
    assert "tool_calls" in block

    assert 'yield "data: [DONE]' in block
    assert "return" in block

    assert "llm_call" not in block
    assert "execute_tool" not in block
    assert "tool_start" not in block
    assert "tool_output" not in block


def test_v9_v8_fail_closed_semantics_remain_preserved() -> None:
    text = (
        "read and verify Brain receipt "
        f"{_V8_RECEIPT}"
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_request_is_ambiguous(
            text
        )
        is True
    )

    assert (
        agent_loop
        ._bilingual_brain_receipt_args_from_text(
            text
        )
        is None
    )

    intent = (
        agent_loop
        ._classify_agent_request(
            [],
            text,
        )
    )

    assert intent["domains"] == set()
    assert intent["low_signal"] is True
