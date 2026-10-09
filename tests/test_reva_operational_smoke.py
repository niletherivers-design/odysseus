"""Repeatable non-destructive REVA operational smoke test.

This test exercises the real Odysseus Bilingual Brain bridge, receipt writer,
receipt reader, result formatter, registry, and receipt-routing helper.

It deliberately does NOT contact a live Bilingual Brain service and does NOT
write into the permanent /app/workspace receipt directory.

All HTTP transport is replaced by a deterministic in-memory fake and all
receipt persistence is confined to pytest's tmp_path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from src.agent_loop import (
    _bilingual_brain_receipt_args_from_text,
)
from src.agent_tools import (
    TOOL_HANDLERS,
)
from src.agent_tools import (
    bilingual_brain_receipts_tool as receipts,
)
from src.agent_tools import (
    bilingual_brain_tool as brain,
)
from src.tool_execution import format_tool_result


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        body: dict[str, Any],
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient replacement used only by this smoke test."""

    calls: list[dict[str, Any]] = []

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        **_: Any,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        return None

    async def post(
        self,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        **_: Any,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "path": path,
                "data": data,
                "headers": headers,
                "json": json,
            }
        )

        if path == "/auth/token":
            assert data is not None
            assert data["username"] == "reva-smoke"
            assert data["password"] == "smoke-password"

            return _FakeResponse(
                status_code=200,
                body={
                    "access_token": "temporary-smoke-token",
                    "token_type": "bearer",
                },
            )

        if path == "/api/brain/run":
            assert headers == {
                "Authorization":
                    "Bearer temporary-smoke-token",
            }

            assert json is not None
            assert json["meshcore_evidence"] is True
            assert json["goal"] == (
                "REVA repeatable operational smoke test"
            )

            return _FakeResponse(
                status_code=200,
                body={
                    "goal":
                        "REVA repeatable operational smoke test",
                    "iterations_requested":
                        1,
                    "iterations_completed":
                        1,
                    "results": [
                        {
                            "iteration":
                                1,
                            "observe": {
                                "observation":
                                    "Synthetic smoke-test observation",
                            },
                            "progress":
                                "Synthetic smoke-test progress",
                            "assumptions": [],
                            "next_step":
                                "No external action",
                            "meshcore_evidence": {
                                "schema":
                                    (
                                        "bilingual-brain."
                                        "meshcore-evidence-summary.v1"
                                    ),
                                "production_authority":
                                    "OFF",
                                "private_keys_destroyed":
                                    True,
                                "synthetic_smoke_test":
                                    True,
                            },
                        },
                    ],
                },
            )

        raise AssertionError(
            f"unexpected fake HTTP path: {path}"
        )


def test_reva_operational_chain_is_repeatable_and_non_destructive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the complete local REVA evidence transport chain."""

    workspace = tmp_path / "workspace"
    receipt_dir = workspace / "receipts"

    workspace.mkdir()

    # Confine BOTH writer and reader to pytest's temporary workspace.
    monkeypatch.setattr(
        brain,
        "_RECEIPT_ROOT",
        workspace,
    )
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        workspace,
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "true",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(receipt_dir),
    )

    # Use deterministic configuration and parsed arguments so this test
    # exercises the operational bridge rather than environment parsing.
    monkeypatch.setattr(
        brain,
        "_configuration",
        lambda: (
            "http://reva-smoke.invalid",
            "reva-smoke",
            "smoke-password",
            30.0,
        ),
    )

    payload = {
        "goal":
            "REVA repeatable operational smoke test",
        "source_lang":
            "English",
        "target_lang":
            "French",
        "iterations":
            1,
        "execute":
            True,
        "horizon_years":
            10,
        "evolve":
            False,
        "seed_genome":
            "SMOKE",
        "meshcore_evidence":
            True,
    }

    monkeypatch.setattr(
        brain,
        "_parse_args",
        lambda _content: dict(payload),
    )

    _FakeAsyncClient.calls = []

    monkeypatch.setattr(
        brain.httpx,
        "AsyncClient",
        _FakeAsyncClient,
    )

    # Registry ownership is part of the operational contract.
    assert (
        TOOL_HANDLERS["bilingual_brain"]
        is brain.handle_bilingual_brain
    )
    assert (
        TOOL_HANDLERS["bilingual_brain_receipts"]
        is receipts.handle_bilingual_brain_receipts
    )

    # Execute the real bridge using only the fake transport.
    result = asyncio.run(
        brain.handle_bilingual_brain(
            "{}",
            {},
        )
    )

    assert "error" not in result
    assert result["iterations_completed"] == 1
    assert len(result["results"]) == 1

    assert len(_FakeAsyncClient.calls) == 2
    assert (
        _FakeAsyncClient.calls[0]["path"]
        == "/auth/token"
    )
    assert (
        _FakeAsyncClient.calls[1]["path"]
        == "/api/brain/run"
    )

    bridge = result[
        "_odysseus_bilingual_brain_bridge"
    ]

    assert bridge["tool"] == "bilingual_brain"
    assert bridge["meshcore_evidence_forced"] is True
    assert bridge["evidence_results"] == 1
    assert bridge["model_claims_auto_verified"] is False
    assert bridge["production_authority"] == "OFF"
    assert bridge["receipt_enabled"] is True

    receipt_path = Path(
        bridge["receipt_path"]
    )

    assert receipt_path.parent == receipt_dir
    assert receipt_path.is_file()

    receipt_bytes = receipt_path.read_bytes()

    receipt_sha256 = hashlib.sha256(
        receipt_bytes
    ).hexdigest()

    assert (
        receipt_sha256
        == bridge["receipt_sha256"]
    )
    assert receipt_path.name.endswith(
        f"-{receipt_sha256[:16]}.json"
    )

    receipt_document = json.loads(
        receipt_bytes
    )

    assert (
        receipt_document["schema"]
        == "odysseus.bilingual-brain-evidence-receipt.v1"
    )

    policy = receipt_document["policy"]

    assert policy["meshcore_evidence_forced"] is True
    assert policy["model_claims_auto_verified"] is False
    assert policy["production_authority"] == "OFF"
    assert policy["private_signing_keys_persisted"] is False

    persisted_bridge = receipt_document[
        "response"
    ][
        "_odysseus_bilingual_brain_bridge"
    ]

    # Critical anti-self-reference property:
    # the receipt is persisted BEFORE its own path/hash are attached
    # to the in-memory response.
    assert persisted_bridge["receipt_enabled"] is False
    assert "receipt_path" not in persisted_bridge
    assert "receipt_sha256" not in persisted_bridge

    serialized = receipt_bytes.decode("utf-8")

    # Transport credentials/token must never enter the receipt.
    assert "smoke-password" not in serialized
    assert "temporary-smoke-token" not in serialized

    # Reader: list.
    listed = asyncio.run(
        receipts.handle_bilingual_brain_receipts(
            json.dumps(
                {
                    "action": "list",
                    "limit": 10,
                }
            ),
            {},
        )
    )

    assert listed["action"] == "list"
    assert listed["total_receipts"] == 1
    assert listed["returned"] == 1
    assert (
        listed["results"][0]["receipt"]
        == receipt_path.name
    )
    assert listed["semantic_claims_verified"] is False
    assert listed["production_authority"] == "OFF"

    # Reader: latest.
    latest = asyncio.run(
        receipts.handle_bilingual_brain_receipts(
            json.dumps(
                {
                    "action": "latest",
                }
            ),
            {},
        )
    )

    assert "error" not in latest
    assert latest["action"] == "latest"
    assert latest["receipt"] == receipt_path.name
    assert latest["receipt_sha256"] == receipt_sha256
    assert latest["filename_sha256_prefix_valid"] is True
    assert latest["receipt_policy_valid"] is True
    assert latest["semantic_claims_verified"] is False
    assert latest["production_authority"] == "OFF"

    # Reader: explicit integrity verification.
    verified = asyncio.run(
        receipts.handle_bilingual_brain_receipts(
            json.dumps(
                {
                    "action":
                        "verify",
                    "receipt":
                        receipt_path.name,
                    "expected_sha256":
                        receipt_sha256,
                }
            ),
            {},
        )
    )

    assert "error" not in verified
    assert verified["action"] == "verify"
    assert verified["receipt"] == receipt_path.name
    assert verified["semantic_claims_verified"] is False
    assert verified["production_authority"] == "OFF"

    verification = verified["results"][0]

    assert verification["sha256"] == receipt_sha256
    assert all(
        verification["checks"].values()
    )

    # Deterministic receipt routing must select read-only inspection.
    assert (
        _bilingual_brain_receipt_args_from_text(
            "show my latest Bilingual Brain receipt"
        )
        == {
            "action": "latest",
        }
    )

    assert (
        _bilingual_brain_receipt_args_from_text(
            "list all my Bilingual Brain receipts"
        )
        == {
            "action": "list",
            "limit": 10,
        }
    )

    # Discussion/analysis about the receipt architecture must NOT be
    # hijacked into receipt reading.
    assert (
        _bilingual_brain_receipt_args_from_text(
            (
                "Analyze with the Bilingual Brain whether the current "
                "evidence-receipt integration preserves a clean "
                "separation between semantic claims and evidence "
                "transport."
            )
        )
        is None
    )

    # Structured result formatting must remain safe for Brain output.
    rendered = format_tool_result(
        "REVA operational smoke",
        result,
    )

    assert isinstance(rendered, str)
    assert "### REVA operational smoke" in rendered
    assert "**results:**" in rendered
    assert "Synthetic smoke-test observation" in rendered

    # The test generated exactly one TEMPORARY receipt.
    assert len(
        list(
            receipt_dir.glob("*.json")
        )
    ) == 1
