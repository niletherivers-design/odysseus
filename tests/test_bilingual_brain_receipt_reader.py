import hashlib
import json
from pathlib import Path

import pytest

from src.agent_tools import bilingual_brain_receipts_tool as receipts


def _document():
    return {
        "schema":
            "odysseus.bilingual-brain-evidence-receipt.v1",
        "created_at":
            "2026-10-08T04:13:41+00:00",
        "request": {
            "goal":
                "receipt-reader-test",
            "meshcore_evidence":
                True,
        },
        "response": {
            "results": [],
        },
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


def _write_receipt(
    directory: Path,
    timestamp: str,
):
    doc = _document()

    raw = (
        json.dumps(
            doc,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        raw
    ).hexdigest()

    name = (
        f"{timestamp}-{digest[:16]}.json"
    )

    path = directory / name
    path.write_bytes(
        raw
    )

    return (
        name,
        digest,
        path,
    )


@pytest.mark.asyncio
async def test_list_receipts_is_newest_first(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        tmp_path,
    )

    directory = (
        tmp_path / "receipts"
    )

    directory.mkdir()

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(directory),
    )

    old_name, _, _ = _write_receipt(
        directory,
        "20261008T040000.000000Z",
    )

    new_name, _, _ = _write_receipt(
        directory,
        "20261008T050000.000000Z",
    )

    result = await receipts.handle_bilingual_brain_receipts(
        json.dumps({
            "action": "list",
            "limit": 10,
        })
    )

    assert result["total_receipts"] == 2

    assert [
        item["receipt"]
        for item in result["results"]
    ] == [
        new_name,
        old_name,
    ]

    assert result["semantic_claims_verified"] is False
    assert result["production_authority"] == "OFF"


@pytest.mark.asyncio
async def test_latest_reads_latest_receipt(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        tmp_path,
    )

    directory = (
        tmp_path / "receipts"
    )

    directory.mkdir()

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(directory),
    )

    _write_receipt(
        directory,
        "20261008T040000.000000Z",
    )

    new_name, digest, _ = _write_receipt(
        directory,
        "20261008T050000.000000Z",
    )

    result = await receipts.handle_bilingual_brain_receipts(
        '{"action":"latest"}'
    )

    assert result["receipt"] == new_name
    assert result["receipt_sha256"] == digest
    assert result["filename_sha256_prefix_valid"] is True
    assert result["receipt_policy_valid"] is True
    assert result["semantic_claims_verified"] is False


@pytest.mark.asyncio
async def test_verify_full_sha256_and_policy(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        tmp_path,
    )

    directory = (
        tmp_path / "receipts"
    )

    directory.mkdir()

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(directory),
    )

    name, digest, _ = _write_receipt(
        directory,
        "20261008T050000.000000Z",
    )

    result = await receipts.handle_bilingual_brain_receipts(
        json.dumps({
            "action":
                "verify",
            "receipt":
                name,
            "expected_sha256":
                digest,
        })
    )

    verification = result[
        "results"
    ][0]

    assert verification["verified"] is True
    assert verification["sha256"] == digest
    assert verification["checks"]["expected_sha256"] is True
    assert verification["semantic_claims_verified"] is False
    assert verification["production_authority"] == "OFF"


@pytest.mark.asyncio
async def test_verify_detects_modified_receipt(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        tmp_path,
    )

    directory = (
        tmp_path / "receipts"
    )

    directory.mkdir()

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(directory),
    )

    name, digest, path = _write_receipt(
        directory,
        "20261008T050000.000000Z",
    )

    path.write_bytes(
        path.read_bytes()
        + b"\n"
    )

    result = await receipts.handle_bilingual_brain_receipts(
        json.dumps({
            "action":
                "verify",
            "receipt":
                name,
            "expected_sha256":
                digest,
        })
    )

    verification = result[
        "results"
    ][0]

    assert verification["verified"] is False
    assert verification["checks"]["filename_sha256_prefix"] is False
    assert verification["checks"]["expected_sha256"] is False


@pytest.mark.asyncio
async def test_rejects_noncanonical_receipt_name(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        receipts,
        "_WORKSPACE_ROOT",
        tmp_path,
    )

    monkeypatch.setenv(
        "BILINGUAL_BRAIN_RECEIPT_DIR",
        str(
            tmp_path / "receipts"
        ),
    )

    result = await receipts.handle_bilingual_brain_receipts(
        json.dumps({
            "action":
                "read",
            "receipt":
                "../../secret.json",
        })
    )

    assert result["exit_code"] == 1
    assert "canonical receipt filename" in result["error"]
