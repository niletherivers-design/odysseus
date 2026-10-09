"""Odysseus -> Bilingual Brain bridge.

This module deliberately contains no MeshCore evidence semantics.

The only evidence-related action performed here is forcing
``meshcore_evidence=True`` on the Bilingual Brain request.  The Brain owns
provenance construction and the MeshCore HTTP adapter owns evidence semantics.

Model-generated factual claims remain UNKNOWN unless separately verified.
PRODUCTION_AUTHORITY remains OFF.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


_TOOL_NAME = "bilingual_brain"


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {
        "error": message,
        "exit_code": 1,
        **extra,
    }


def _parse_args(content: str) -> dict[str, Any]:
    try:
        args = json.loads(
            content
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise ValueError(
            "bilingual_brain requires JSON arguments"
        ) from exc

    if not isinstance(
        args,
        dict,
    ):
        raise ValueError(
            "bilingual_brain arguments must be a JSON object"
        )

    allowed = {
        "goal",
        "source_lang",
        "target_lang",
        "iterations",
        "execute",
        "horizon_years",
        "evolve",
        "seed_genome",
    }

    unknown = (
        set(args)
        - allowed
    )

    if unknown:
        raise ValueError(
            "unsupported bilingual_brain argument(s): "
            + ", ".join(
                sorted(
                    unknown
                )
            )
        )

    goal = args.get(
        "goal"
    )

    if (
        not isinstance(
            goal,
            str,
        )
        or not goal.strip()
    ):
        raise ValueError(
            "goal must be a non-empty string"
        )

    iterations = args.get(
        "iterations",
        1,
    )

    if (
        isinstance(
            iterations,
            bool,
        )
        or not isinstance(
            iterations,
            int,
        )
        or not (
            1
            <= iterations
            <= 10
        )
    ):
        raise ValueError(
            "iterations must be an integer from 1 to 10"
        )

    horizon_years = args.get(
        "horizon_years",
        10,
    )

    if (
        isinstance(
            horizon_years,
            bool,
        )
        or not isinstance(
            horizon_years,
            int,
        )
        or not (
            0
            <= horizon_years
            <= 1000
        )
    ):
        raise ValueError(
            "horizon_years must be an integer from 0 to 1000"
        )

    execute = args.get(
        "execute",
        True,
    )

    evolve = args.get(
        "evolve",
        False,
    )

    if not isinstance(
        execute,
        bool,
    ):
        raise ValueError(
            "execute must be boolean"
        )

    if not isinstance(
        evolve,
        bool,
    ):
        raise ValueError(
            "evolve must be boolean"
        )

    source_lang = args.get(
        "source_lang",
        "English",
    )

    target_lang = args.get(
        "target_lang",
        "English",
    )

    seed_genome = args.get(
        "seed_genome",
        "A",
    )

    for name, value in (
        (
            "source_lang",
            source_lang,
        ),
        (
            "target_lang",
            target_lang,
        ),
        (
            "seed_genome",
            seed_genome,
        ),
    ):
        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{name} must be a non-empty string"
            )

    return {
        "goal":
            goal.strip(),
        "source_lang":
            source_lang.strip(),
        "target_lang":
            target_lang.strip(),
        "iterations":
            iterations,
        "execute":
            execute,
        "horizon_years":
            horizon_years,
        "evolve":
            evolve,
        "seed_genome":
            seed_genome.strip(),

        # This is intentionally not user/model configurable.
        "meshcore_evidence":
            True,
    }


def _configuration() -> tuple[
    str,
    str,
    str,
    float,
]:
    base_url = os.getenv(
        "BILINGUAL_BRAIN_URL",
        "http://host.docker.internal:8110",
    ).strip().rstrip(
        "/"
    )

    username = os.getenv(
        "BILINGUAL_BRAIN_USERNAME",
        "",
    ).strip()

    password = os.getenv(
        "BILINGUAL_BRAIN_PASSWORD",
        "",
    )

    timeout_raw = os.getenv(
        "BILINGUAL_BRAIN_TIMEOUT_SECONDS",
        "720",
    )

    parsed = urlparse(
        base_url
    )

    if (
        parsed.scheme
        not in {
            "http",
            "https",
        }
        or not parsed.netloc
    ):
        raise ValueError(
            "BILINGUAL_BRAIN_URL is invalid"
        )

    if not username:
        raise ValueError(
            "BILINGUAL_BRAIN_USERNAME is not configured"
        )

    if not password:
        raise ValueError(
            "BILINGUAL_BRAIN_PASSWORD is not configured"
        )

    try:
        timeout = float(
            timeout_raw
        )
    except ValueError as exc:
        raise ValueError(
            "BILINGUAL_BRAIN_TIMEOUT_SECONDS is invalid"
        ) from exc

    if not (
        1.0
        <= timeout
        <= 1800.0
    ):
        raise ValueError(
            "BILINGUAL_BRAIN_TIMEOUT_SECONDS must be between 1 and 1800"
        )

    return (
        base_url,
        username,
        password,
        timeout,
    )


_RECEIPT_ROOT = Path("/app/workspace")


def _receipt_settings() -> tuple[bool, Path]:
    """Return receipt enablement and a path confined to /app/workspace."""
    raw_enabled = os.getenv(
        "BILINGUAL_BRAIN_SAVE_RECEIPTS",
        "0",
    ).strip().lower()

    if raw_enabled in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        enabled = False
    elif raw_enabled in {
        "1",
        "true",
        "yes",
        "on",
    }:
        enabled = True
    else:
        raise ValueError(
            "BILINGUAL_BRAIN_SAVE_RECEIPTS must be "
            "0/1, false/true, no/yes, or off/on"
        )

    raw_dir = os.getenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        "/app/workspace/bilingual-brain-receipts",
    ).strip()

    if not raw_dir:
        raise ValueError(
            "BILINGUAL_BRAIN_RECEIPT_DIR must not be empty"
        )

    root = _RECEIPT_ROOT.resolve()
    receipt_dir = Path(raw_dir).expanduser().resolve()

    if (
        receipt_dir != root
        and root not in receipt_dir.parents
    ):
        raise ValueError(
            "BILINGUAL_BRAIN_RECEIPT_DIR must remain "
            "inside /app/workspace"
        )

    return enabled, receipt_dir


def _write_receipt_sync(
    *,
    request_payload: dict[str, Any],
    response_body: dict[str, Any],
) -> tuple[str, str] | None:
    """Persist one canonical receipt without credentials or private keys."""
    enabled, receipt_dir = _receipt_settings()

    if not enabled:
        return None

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    receipt = {
        "schema":
            "odysseus.bilingual-brain-evidence-receipt.v1",
        "created_at":
            created_at,
        "request":
            request_payload,
        "response":
            response_body,
        "policy": {
            "meshcore_evidence_forced":
                True,
            "model_claims_auto_verified":
                False,
            "production_authority":
                "OFF",
            "private_signing_keys_persisted":
                False,
            "posix_mode_requested":
                "0600",
            "effective_access_control":
                "host_filesystem",
        },
    }

    encoded = (
        json.dumps(
            receipt,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    digest = hashlib.sha256(
        encoded
    ).hexdigest()

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )

    receipt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = receipt_dir / (
        f"{timestamp}-{digest[:16]}.json"
    )

    temp_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".receipt-",
            suffix=".tmp",
            dir=receipt_dir,
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(
                encoded
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_name,
            final_path,
        )

        # Request restrictive POSIX permissions where the backing filesystem
        # supports them. Windows-backed WSL/DrvFS/v9fs mounts may expose broad
        # synthetic POSIX mode bits while Windows ACLs remain authoritative.
        os.chmod(
            final_path,
            0o600,
        )
    except Exception:
        if temp_name:
            try:
                os.unlink(
                    temp_name
                )
            except OSError:
                pass
        raise

    return (
        str(final_path),
        digest,
    )


async def handle_bilingual_brain(
    content: str,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Bilingual Brain request with MeshCore evidence forced on."""

    # Receipt persistence uses a fixed, confined persistent workspace root;
    # it does not depend on model-supplied paths or active-workspace state.
    del ctx

    try:
        payload = _parse_args(
            content
        )

        (
            base_url,
            username,
            password,
            timeout,
        ) = _configuration()

    except ValueError as exc:
        return _error(
            str(exc)
        )

    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        ) as client:
            token_response = await client.post(
                "/auth/token",
                data={
                    "username":
                        username,
                    "password":
                        password,
                    "grant_type":
                        "password",
                    "scope":
                        "",
                },
            )

            if (
                token_response.status_code
                != 200
            ):
                return _error(
                    "Bilingual Brain authentication failed",
                    status_code=(
                        token_response.status_code
                    ),
                )

            token_body = (
                token_response.json()
            )

            access_token = (
                token_body.get(
                    "access_token"
                )
            )

            if (
                not isinstance(
                    access_token,
                    str,
                )
                or not access_token
            ):
                return _error(
                    "Bilingual Brain authentication response "
                    "did not contain an access token"
                )

            run_response = await client.post(
                "/api/brain/run",
                headers={
                    "Authorization":
                        "Bearer "
                        + access_token,
                },
                json=payload,
            )

    except httpx.TimeoutException:
        return _error(
            "Bilingual Brain request timed out"
        )

    except httpx.HTTPError as exc:
        return _error(
            "Bilingual Brain HTTP request failed",
            detail=str(
                exc
            ),
        )

    if (
        run_response.status_code
        != 200
    ):
        body_preview = (
            run_response.text[
                :4000
            ]
        )

        return _error(
            "Bilingual Brain run failed",
            status_code=(
                run_response.status_code
            ),
            response=body_preview,
        )

    try:
        body = (
            run_response.json()
        )
    except ValueError:
        return _error(
            "Bilingual Brain returned invalid JSON"
        )

    if not isinstance(
        body,
        dict,
    ):
        return _error(
            "Bilingual Brain returned an invalid response shape"
        )

    results = body.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        return _error(
            "Bilingual Brain response has no results list"
        )

    evidence_count = 0

    for item in results:
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get(
                    "meshcore_evidence"
                ),
                dict,
            )
        ):
            evidence_count += 1

    expected_count = body.get(
        "iterations_completed"
    )

    if (
        isinstance(
            expected_count,
            int,
        )
        and expected_count
        != evidence_count
    ):
        return _error(
            "Bilingual Brain result is missing MeshCore evidence",
            iterations_completed=(
                expected_count
            ),
            evidence_results=(
                evidence_count
            ),
        )

    # Add only bridge metadata. Do not reinterpret or promote Brain/MeshCore
    # evidence states in Odysseus.
    body[
        "_odysseus_bilingual_brain_bridge"
    ] = {
        "tool":
            _TOOL_NAME,
        "meshcore_evidence_forced":
            True,
        "evidence_results":
            evidence_count,
        "model_claims_auto_verified":
            False,
        "production_authority":
            "OFF",
        "receipt_enabled":
            False,
    }

    bridge = body[
        "_odysseus_bilingual_brain_bridge"
    ]

    try:
        receipt_result = _write_receipt_sync(
            request_payload=payload,
            response_body=body,
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        # Receipt capture is observational only. Never convert a valid Brain /
        # MeshCore result into a failed tool execution because local persistence
        # was unavailable or misconfigured.
        bridge[
            "receipt_error"
        ] = str(
            exc
        )
    else:
        if receipt_result is not None:
            (
                receipt_path,
                receipt_sha256,
            ) = receipt_result

            bridge[
                "receipt_enabled"
            ] = True
            bridge[
                "receipt_path"
            ] = receipt_path
            bridge[
                "receipt_sha256"
            ] = receipt_sha256

    return body
