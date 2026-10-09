"""Read-only access to persisted Bilingual Brain evidence receipts.

Receipts are workspace content and therefore remain untrusted as semantic
evidence. This tool can inspect and hash them, but it does not generate new
MeshCore evidence, invoke the Brain model, or promote factual claims.

PRODUCTION_AUTHORITY remains OFF.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


_TOOL_NAME = "bilingual_brain_receipts"

_WORKSPACE_ROOT = Path("/app/workspace")

_DEFAULT_RECEIPT_DIR = (
    "/app/workspace/bilingual-brain-receipts"
)

_RECEIPT_NAME_RE = re.compile(
    r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{16}\.json$"
)

_SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)

_RECEIPT_SCHEMA = (
    "odysseus.bilingual-brain-evidence-receipt.v1"
)


def _error(
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "error": message,
        "exit_code": 1,
        **extra,
    }


def _receipt_dir() -> Path:
    root = _WORKSPACE_ROOT.resolve()

    raw = os.getenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        _DEFAULT_RECEIPT_DIR,
    ).strip()

    if not raw:
        raise ValueError(
            "BILINGUAL_BRAIN_RECEIPT_DIR must not be empty"
        )

    directory = Path(
        raw
    ).expanduser().resolve()

    if (
        directory != root
        and root not in directory.parents
    ):
        raise ValueError(
            "receipt directory must remain inside /app/workspace"
        )

    return directory


def _parse_args(
    content: str,
) -> dict[str, Any]:
    try:
        args = json.loads(
            content or "{}"
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise ValueError(
            "bilingual_brain_receipts requires JSON arguments"
        ) from exc

    if not isinstance(
        args,
        dict,
    ):
        raise ValueError(
            "bilingual_brain_receipts arguments must be a JSON object"
        )

    allowed = {
        "action",
        "receipt",
        "limit",
        "expected_sha256",
    }

    unknown = (
        set(args)
        - allowed
    )

    if unknown:
        raise ValueError(
            "unsupported bilingual_brain_receipts argument(s): "
            + ", ".join(
                sorted(
                    unknown
                )
            )
        )

    action = args.get(
        "action",
        "list",
    )

    if action not in {
        "list",
        "latest",
        "read",
        "verify",
    }:
        raise ValueError(
            "action must be one of: list, latest, read, verify"
        )

    limit = args.get(
        "limit",
        10,
    )

    if (
        isinstance(
            limit,
            bool,
        )
        or not isinstance(
            limit,
            int,
        )
        or not (
            1
            <= limit
            <= 50
        )
    ):
        raise ValueError(
            "limit must be an integer from 1 to 50"
        )

    receipt = args.get(
        "receipt"
    )

    if receipt is not None:
        if (
            not isinstance(
                receipt,
                str,
            )
            or not _RECEIPT_NAME_RE.fullmatch(
                receipt
            )
        ):
            raise ValueError(
                "receipt must be a canonical receipt filename"
            )

    if (
        action
        in {
            "read",
            "verify",
        }
        and receipt is None
    ):
        raise ValueError(
            f"receipt is required for action={action}"
        )

    expected_sha256 = args.get(
        "expected_sha256"
    )

    if expected_sha256 is not None:
        if not isinstance(
            expected_sha256,
            str,
        ):
            raise ValueError(
                "expected_sha256 must be a SHA-256 string"
            )

        expected_sha256 = (
            expected_sha256.strip().lower()
        )

        if not _SHA256_RE.fullmatch(
            expected_sha256
        ):
            raise ValueError(
                "expected_sha256 must be exactly 64 hexadecimal characters"
            )

    return {
        "action":
            action,
        "receipt":
            receipt,
        "limit":
            limit,
        "expected_sha256":
            expected_sha256,
    }


def _receipt_names(
    directory: Path,
) -> list[str]:
    if not directory.exists():
        return []

    if not directory.is_dir():
        raise ValueError(
            "receipt path exists but is not a directory"
        )

    names = [
        entry.name
        for entry in directory.iterdir()
        if (
            entry.is_file()
            and _RECEIPT_NAME_RE.fullmatch(
                entry.name
            )
        )
    ]

    return sorted(
        names,
        reverse=True,
    )


def _resolve_receipt(
    directory: Path,
    name: str,
) -> Path:
    if not _RECEIPT_NAME_RE.fullmatch(
        name
    ):
        raise ValueError(
            "invalid receipt filename"
        )

    try:
        path = (
            directory
            / name
        ).resolve(
            strict=True
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"receipt not found: {name}"
        ) from exc

    resolved_directory = (
        directory.resolve()
    )

    if (
        path.parent
        != resolved_directory
    ):
        raise ValueError(
            "receipt resolves outside the receipt directory"
        )

    if not path.is_file():
        raise ValueError(
            f"receipt is not a regular file: {name}"
        )

    return path


def _read_receipt(
    path: Path,
) -> tuple[
    bytes,
    dict[str, Any],
]:
    raw = path.read_bytes()

    if len(
        raw
    ) > 8_000_000:
        raise ValueError(
            "receipt exceeds the 8 MB read limit"
        )

    try:
        document = json.loads(
            raw
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            "receipt is not valid JSON"
        ) from exc

    if not isinstance(
        document,
        dict,
    ):
        raise ValueError(
            "receipt JSON must be an object"
        )

    return (
        raw,
        document,
    )


def _contains_sensitive_key(
    value: Any,
) -> bool:
    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            if str(
                key
            ).lower() in {
                "access_token",
                "password",
                "private_key",
                "private_signing_key",
            }:
                return True

            if _contains_sensitive_key(
                child
            ):
                return True

    elif isinstance(
        value,
        list,
    ):
        return any(
            _contains_sensitive_key(
                item
            )
            for item in value
        )

    return False


def _verification(
    *,
    path: Path,
    raw: bytes,
    document: dict[str, Any],
    expected_sha256: str | None,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        raw
    ).hexdigest()

    filename_prefix = (
        path.stem.rsplit(
            "-",
            1,
        )[-1]
    )

    policy = document.get(
        "policy"
    )

    policy_is_object = isinstance(
        policy,
        dict,
    )

    checks = {
        "schema":
            document.get(
                "schema"
            )
            == _RECEIPT_SCHEMA,
        "filename_sha256_prefix":
            digest.startswith(
                filename_prefix
            ),
        "policy_object":
            policy_is_object,
        "meshcore_evidence_forced":
            (
                policy_is_object
                and policy.get(
                    "meshcore_evidence_forced"
                )
                is True
            ),
        "model_claims_auto_verified_false":
            (
                policy_is_object
                and policy.get(
                    "model_claims_auto_verified"
                )
                is False
            ),
        "production_authority_off":
            (
                policy_is_object
                and policy.get(
                    "production_authority"
                )
                == "OFF"
            ),
        "private_signing_keys_persisted_false":
            (
                policy_is_object
                and policy.get(
                    "private_signing_keys_persisted"
                )
                is False
            ),
        "sensitive_keys_absent":
            not _contains_sensitive_key(
                document
            ),
        "private_key_material_absent":
            b"BEGIN PRIVATE KEY"
            not in raw,
    }

    if expected_sha256 is not None:
        checks[
            "expected_sha256"
        ] = (
            digest
            == expected_sha256
        )

    return {
        "verified":
            all(
                checks.values()
            ),
        "sha256":
            digest,
        "filename_sha256_prefix":
            filename_prefix,
        "expected_sha256":
            expected_sha256,
        "full_expected_sha256_supplied":
            expected_sha256
            is not None,
        "checks":
            checks,
        "semantic_claims_verified":
            False,
        "production_authority":
            "OFF",
    }


async def handle_bilingual_brain_receipts(
    content: str,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List, read, or integrity-check persisted evidence receipts."""

    del ctx

    try:
        args = _parse_args(
            content
        )

        directory = _receipt_dir()

        action = args[
            "action"
        ]

        if action == "list":
            names = _receipt_names(
                directory
            )

            selected = names[
                :args["limit"]
            ]

            results = []

            for name in selected:
                path = _resolve_receipt(
                    directory,
                    name,
                )

                stat = path.stat()

                results.append(
                    {
                        "receipt":
                            name,
                        "size_bytes":
                            stat.st_size,
                    }
                )

            return {
                "results":
                    results,
                "action":
                    "list",
                "total_receipts":
                    len(names),
                "returned":
                    len(results),
                "receipt_directory":
                    str(directory),
                "semantic_claims_verified":
                    False,
                "production_authority":
                    "OFF",
            }

        if action == "latest":
            names = _receipt_names(
                directory
            )

            if not names:
                return _error(
                    "no bilingual Brain receipts exist"
                )

            name = names[
                0
            ]

        else:
            name = args[
                "receipt"
            ]

        path = _resolve_receipt(
            directory,
            name,
        )

        raw, document = _read_receipt(
            path
        )

        verification = _verification(
            path=path,
            raw=raw,
            document=document,
            expected_sha256=(
                args[
                    "expected_sha256"
                ]
                if action
                == "verify"
                else None
            ),
        )

        if action == "verify":
            return {
                "results": [
                    verification
                ],
                "action":
                    "verify",
                "receipt":
                    name,
                "semantic_claims_verified":
                    False,
                "production_authority":
                    "OFF",
            }

        return {
            "results": [
                document
            ],
            "action":
                action,
            "receipt":
                name,
            "receipt_sha256":
                verification[
                    "sha256"
                ],
            "filename_sha256_prefix_valid":
                verification[
                    "checks"
                ][
                    "filename_sha256_prefix"
                ],
            "receipt_policy_valid":
                all(
                    value
                    for key, value
                    in verification[
                        "checks"
                    ].items()
                    if key
                    not in {
                        "filename_sha256_prefix",
                    }
                ),
            "semantic_claims_verified":
                False,
            "production_authority":
                "OFF",
        }

    except (
        OSError,
        ValueError,
    ) as exc:
        return _error(
            str(
                exc
            )
        )
