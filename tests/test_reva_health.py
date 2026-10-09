"""Tests for the read-only REVA runtime health tool."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src import tool_capabilities
from src.agent_tools import TOOL_HANDLERS
from src.agent_tools import bilingual_brain_receipts_tool as receipts
from src.agent_tools import bilingual_brain_tool as brain
from src.agent_tools import reva_health_tool as health
from src.tool_capabilities import ResultIntegrity, ToolEffect
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS


def _checks_by_name(
    result: dict,
) -> dict[str, dict]:
    return {
        item["name"]: item
        for item in result["results"]
    }


class _ForbiddenNetworkClient:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise AssertionError(
            "reva_health must not instantiate an HTTP client"
        )


def test_reva_health_reports_local_health_without_network_or_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    receipt_dir = workspace / "receipts"

    workspace.mkdir()

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
        "BILINGUAL_BRAIN_URL",
        "http://reva-health.invalid:8110",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_USERNAME",
        "health-user",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_PASSWORD",
        "health-secret-password",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_TIMEOUT_SECONDS",
        "30",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "true",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(receipt_dir),
    )

    monkeypatch.setattr(
        brain.httpx,
        "AsyncClient",
        _ForbiddenNetworkClient,
    )

    receipt_result = brain._write_receipt_sync(
        request_payload={
            "goal":
                "temporary health test",
            "meshcore_evidence":
                True,
        },
        response_body={
            "results": [],
            "_odysseus_bilingual_brain_bridge": {
                "tool":
                    "bilingual_brain",
                "meshcore_evidence_forced":
                    True,
                "evidence_results":
                    0,
                "model_claims_auto_verified":
                    False,
                "production_authority":
                    "OFF",
                "receipt_enabled":
                    False,
            },
        },
    )

    assert receipt_result is not None

    result = asyncio.run(
        health.handle_reva_health(
            "{}",
            {},
        )
    )

    assert "error" not in result
    assert result["service"] == "reva"
    assert result["overall"] == "ok"
    assert result["network_access"] is False
    assert result["brain_executed"] is False
    assert result["model_executed"] is False
    assert result["receipt_generated"] is False
    assert result["semantic_claims_verified"] is False
    assert result["production_authority"] == "OFF"
    assert result["secret_values_exposed"] is False

    checks = _checks_by_name(
        result
    )

    assert checks["tool_registry"]["status"] == "ok"
    assert checks["brain_configuration"]["status"] == "ok"
    assert checks["receipt_configuration"]["status"] == "ok"
    assert checks["latest_receipt_integrity"]["status"] == "ok"
    assert checks["meshcore_policy_invariants"]["status"] == "ok"

    assert (
        checks["latest_receipt_integrity"]["meta"][
            "semantic_claims_verified"
        ]
        is False
    )

    assert all(
        checks["latest_receipt_integrity"]["meta"][
            "checks"
        ].values()
    )

    assert all(
        checks["meshcore_policy_invariants"]["meta"][
            "checks"
        ].values()
    )

    serialized = json.dumps(
        result,
        sort_keys=True,
    )

    assert "health-secret-password" not in serialized
    assert "health-user" not in serialized
    assert "reva-health.invalid" not in serialized

    assert (
        TOOL_HANDLERS["reva_health"]
        is health.handle_reva_health
    )

    names = {
        schema.get(
            "function",
            {},
        ).get(
            "name"
        )
        for schema in FUNCTION_TOOL_SCHEMAS
    }

    assert "reva_health" in names

    capabilities = (
        tool_capabilities._REGISTRY[
            "reva_health"
        ]
    )

    assert (
        ToolEffect.READ_WORKSPACE
        in capabilities.effects
    )
    assert (
        capabilities.result_integrity
        is ResultIntegrity.WORKSPACE_UNTRUSTED
    )


def test_reva_health_degrades_cleanly_when_brain_credentials_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

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
        "BILINGUAL_BRAIN_URL",
        "http://reva-health.invalid:8110",
    )
    monkeypatch.delenv(
        "BILINGUAL_BRAIN_USERNAME",
        raising=False,
    )
    monkeypatch.delenv(
        "BILINGUAL_BRAIN_PASSWORD",
        raising=False,
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_TIMEOUT_SECONDS",
        "30",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "false",
    )
    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(
            workspace / "receipts"
        ),
    )

    monkeypatch.setattr(
        brain.httpx,
        "AsyncClient",
        _ForbiddenNetworkClient,
    )

    result = asyncio.run(
        health.handle_reva_health(
            "{}",
            {},
        )
    )

    assert "error" not in result
    assert result["overall"] == "degraded"
    assert result["network_access"] is False
    assert result["brain_executed"] is False
    assert result["model_executed"] is False
    assert result["receipt_generated"] is False
    assert result["production_authority"] == "OFF"

    checks = _checks_by_name(
        result
    )

    assert (
        checks["brain_configuration"]["status"]
        == "degraded"
    )
    assert (
        checks["brain_configuration"]["meta"][
            "username_configured"
        ]
        is False
    )
    assert (
        checks["brain_configuration"]["meta"][
            "password_configured"
        ]
        is False
    )

    assert (
        checks["receipt_configuration"]["status"]
        == "disabled"
    )
    assert (
        checks["latest_receipt_integrity"]["status"]
        == "disabled"
    )
    assert (
        checks["meshcore_policy_invariants"]["status"]
        == "disabled"
    )


def test_reva_health_rejects_arguments() -> None:
    result = asyncio.run(
        health.handle_reva_health(
            '{"network": true}',
            {},
        )
    )

    assert result["exit_code"] == 1
    assert (
        "does not accept arguments"
        in result["error"]
    )
