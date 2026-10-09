"""Read-only REVA runtime health and self-diagnostic tool.

This diagnostic deliberately performs no network access, does not invoke the
Bilingual Brain or any model, and never creates evidence receipts.

Receipt inspection is local and observational only. Receipt-derived state is
workspace-untrusted and never semantically verifies model-generated claims.
PRODUCTION_AUTHORITY remains OFF.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import bilingual_brain_receipts_tool as receipts
from . import bilingual_brain_tool as brain


_TOOL_NAME = "reva_health"


def _error(
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "error": message,
        "exit_code": 1,
        **extra,
    }


def _parse_args(
    content: str,
) -> dict[str, Any]:
    raw = str(content or "").strip()

    if not raw:
        return {}

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "reva_health requires JSON arguments"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            "reva_health arguments must be a JSON object"
        )

    if value:
        raise ValueError(
            "reva_health does not accept arguments"
        )

    return value


def _check(
    name: str,
    status: str,
    detail: str,
    **meta: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "meta": meta,
    }


def _inside(
    path: Path,
    root: Path,
) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()

    return (
        resolved_path == resolved_root
        or resolved_root in resolved_path.parents
    )


def _overall(
    checks: list[dict[str, Any]],
) -> str:
    for item in checks:
        if item.get("status") == "degraded":
            return "degraded"

    return "ok"


def _brain_configuration_check() -> dict[str, Any]:
    url_available = bool(
        os.getenv(
            "BILINGUAL_BRAIN_URL",
            "http://host.docker.internal:8110",
        ).strip()
    )
    username_configured = bool(
        os.getenv(
            "BILINGUAL_BRAIN_USERNAME",
            "",
        ).strip()
    )
    password_configured = bool(
        os.getenv(
            "BILINGUAL_BRAIN_PASSWORD",
            "",
        )
    )
    timeout_available = bool(
        os.getenv(
            "BILINGUAL_BRAIN_TIMEOUT_SECONDS",
            "720",
        ).strip()
    )

    try:
        # Reuse the canonical validation contract, but deliberately discard
        # every returned value so credentials and endpoint details never enter
        # the diagnostic response.
        brain._configuration()
    except ValueError as exc:
        return _check(
            "brain_configuration",
            "degraded",
            str(exc),
            url_available=url_available,
            username_configured=username_configured,
            password_configured=password_configured,
            timeout_available=timeout_available,
            secret_values_exposed=False,
        )

    return _check(
        "brain_configuration",
        "ok",
        "Bilingual Brain configuration is structurally valid.",
        url_available=url_available,
        username_configured=username_configured,
        password_configured=password_configured,
        timeout_available=timeout_available,
        secret_values_exposed=False,
    )


def _registry_check() -> dict[str, Any]:
    # Delayed import avoids a package-initialization cycle.
    from src import agent_tools

    handlers = getattr(
        agent_tools,
        "TOOL_HANDLERS",
        {},
    )

    expected = {
        "bilingual_brain":
            brain.handle_bilingual_brain,
        "bilingual_brain_receipts":
            receipts.handle_bilingual_brain_receipts,
        "reva_health":
            handle_reva_health,
    }

    registered = {
        name:
            handlers.get(name) is handler
        for name, handler in expected.items()
    }

    healthy = all(
        registered.values()
    )

    return _check(
        "tool_registry",
        "ok" if healthy else "degraded",
        (
            "REVA handlers are registered correctly."
            if healthy
            else
            "One or more REVA handlers are not registered correctly."
        ),
        registered=registered,
    )


def _receipt_configuration_check() -> tuple[
    dict[str, Any],
    bool | None,
    Path | None,
]:
    try:
        enabled, writer_dir = (
            brain._receipt_settings()
        )
        reader_dir = (
            receipts._receipt_dir()
        )
    except ValueError as exc:
        return (
            _check(
                "receipt_configuration",
                "degraded",
                str(exc),
                secret_values_exposed=False,
            ),
            None,
            None,
        )

    writer_root = brain._RECEIPT_ROOT.resolve()
    reader_root = receipts._WORKSPACE_ROOT.resolve()

    writer_confined = _inside(
        writer_dir,
        writer_root,
    )
    reader_confined = _inside(
        reader_dir,
        reader_root,
    )
    directories_match = (
        writer_dir.resolve()
        == reader_dir.resolve()
    )

    healthy = (
        writer_confined
        and reader_confined
        and directories_match
    )

    if not healthy:
        status = "degraded"
        detail = (
            "Receipt writer/reader configuration is inconsistent."
        )
    elif enabled:
        status = "ok"
        detail = (
            "Receipt capture is enabled and confined to the workspace."
        )
    else:
        status = "disabled"
        detail = (
            "Receipt capture is disabled; read-only diagnostics remain available."
        )

    return (
        _check(
            "receipt_configuration",
            status,
            detail,
            enabled=enabled,
            writer_confined=writer_confined,
            reader_confined=reader_confined,
            writer_reader_match=directories_match,
            directory_exists=reader_dir.exists(),
            secret_values_exposed=False,
        ),
        enabled,
        reader_dir,
    )


def _latest_receipt_checks(
    *,
    enabled: bool | None,
    directory: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    if directory is None:
        unavailable = _check(
            "latest_receipt_integrity",
            "degraded",
            "Receipt directory configuration is unavailable.",
            receipt_present=False,
        )
        policy = _check(
            "meshcore_policy_invariants",
            "degraded",
            "Receipt policy cannot be inspected.",
            receipt_present=False,
        )
        return unavailable, policy

    try:
        names = receipts._receipt_names(
            directory
        )
    except (OSError, ValueError) as exc:
        unavailable = _check(
            "latest_receipt_integrity",
            "degraded",
            str(exc),
            receipt_present=False,
        )
        policy = _check(
            "meshcore_policy_invariants",
            "degraded",
            "Receipt policy cannot be inspected.",
            receipt_present=False,
        )
        return unavailable, policy

    if not names:
        status = (
            "degraded"
            if enabled
            else
            "disabled"
        )

        integrity = _check(
            "latest_receipt_integrity",
            status,
            (
                "Receipt capture is enabled but no receipt is available."
                if enabled
                else
                "No receipt is available and receipt capture is disabled."
            ),
            receipt_present=False,
        )

        policy = _check(
            "meshcore_policy_invariants",
            status,
            (
                "No persisted receipt is available for policy inspection."
            ),
            receipt_present=False,
        )

        return integrity, policy

    name = names[0]

    try:
        path = receipts._resolve_receipt(
            directory,
            name,
        )
        raw, document = receipts._read_receipt(
            path
        )
        verification = receipts._verification(
            path=path,
            raw=raw,
            document=document,
            expected_sha256=None,
        )
    except (OSError, ValueError) as exc:
        integrity = _check(
            "latest_receipt_integrity",
            "degraded",
            str(exc),
            receipt_present=True,
            receipt=name,
        )

        policy = _check(
            "meshcore_policy_invariants",
            "degraded",
            "Latest receipt could not be validated.",
            receipt_present=True,
            receipt=name,
        )

        return integrity, policy

    checks = dict(
        verification.get(
            "checks",
            {},
        )
    )

    verified = bool(
        verification.get(
            "verified"
        )
    )

    integrity = _check(
        "latest_receipt_integrity",
        "ok" if verified else "degraded",
        (
            "Latest receipt passed local integrity/security-policy checks."
            if verified
            else
            "Latest receipt failed one or more local integrity/security-policy checks."
        ),
        receipt_present=True,
        receipt=name,
        sha256=verification.get(
            "sha256"
        ),
        checks=checks,
        semantic_claims_verified=False,
    )

    invariant_names = (
        "schema",
        "policy_object",
        "meshcore_evidence_forced",
        "model_claims_auto_verified_false",
        "production_authority_off",
        "private_signing_keys_persisted_false",
        "sensitive_keys_absent",
        "private_key_material_absent",
    )

    invariant_checks = {
        key:
            checks.get(key) is True
        for key in invariant_names
    }

    invariant_ok = all(
        invariant_checks.values()
    )

    policy = _check(
        "meshcore_policy_invariants",
        "ok" if invariant_ok else "degraded",
        (
            "Latest receipt preserves REVA/MeshCore policy invariants."
            if invariant_ok
            else
            "Latest receipt violates one or more REVA/MeshCore policy invariants."
        ),
        receipt_present=True,
        receipt=name,
        checks=invariant_checks,
        semantic_claims_verified=False,
    )

    return integrity, policy


async def handle_reva_health(
    content: str,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return local REVA health without network/model execution or writes."""

    del ctx

    try:
        _parse_args(
            content
        )
    except ValueError as exc:
        return _error(
            str(exc)
        )

    checks: list[dict[str, Any]] = []

    checks.append(
        _registry_check()
    )

    checks.append(
        _brain_configuration_check()
    )

    (
        receipt_configuration,
        receipt_enabled,
        receipt_directory,
    ) = _receipt_configuration_check()

    checks.append(
        receipt_configuration
    )

    (
        receipt_integrity,
        policy_invariants,
    ) = _latest_receipt_checks(
        enabled=receipt_enabled,
        directory=receipt_directory,
    )

    checks.append(
        receipt_integrity
    )
    checks.append(
        policy_invariants
    )

    overall = _overall(
        checks
    )

    return {
        "results": checks,
        "service": "reva",
        "overall": overall,
        "network_access": False,
        "brain_executed": False,
        "model_executed": False,
        "receipt_generated": False,
        "semantic_claims_verified": False,
        "production_authority": "OFF",
        "secret_values_exposed": False,
    }
