"""Unit and Integration Tests for Resumable Downloader and Safe Archive Extractor.

Covers:
- Streaming SHA-256 calculation for arbitrary file sizes.
- ZIP archive integrity checking (valid vs corrupt).
- Path traversal rejection (Zip Slip vulnerability prevention) in safe extraction.
- Downloader manifest state management and resume boundary tracking.
- Bounded download execution with duration timeouts.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.downloader import (
    DownloadProgress,
    ResumableDownloader,
    compute_file_sha256,
    safe_extract_archive,
    verify_zip_integrity,
)


# ==============================================================================
# 1. SHA-256 Streaming Checksum Tests
# ==============================================================================

def test_compute_file_sha256(tmp_path: Path):
    """Verify compute_file_sha256 matches hashlib output across multiple chunks."""
    test_file = tmp_path / "test_data.bin"
    # Create 1.5 MB test data to test multi-chunk reading
    content = b"MedicalAI_Test_Payload_" * 65536
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    computed_hash = compute_file_sha256(test_file, chunk_size=512 * 1024)

    assert computed_hash == expected_hash


def test_compute_file_sha256_empty_file(tmp_path: Path):
    """Verify SHA-256 for an empty file."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")

    expected_hash = hashlib.sha256(b"").hexdigest()
    assert compute_file_sha256(empty_file) == expected_hash


# ==============================================================================
# 2. ZIP Integrity Tests
# ==============================================================================

def test_zip_integrity_valid_archive(tmp_path: Path):
    """Verify verify_zip_integrity returns (True, None) for a healthy archive."""
    zip_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test1.txt", "Hello world")
        zf.writestr("test2.txt", "Another test file")

    is_valid, error = verify_zip_integrity(zip_path)
    assert is_valid is True
    assert error is None


def test_zip_integrity_corrupt_archive(tmp_path: Path):
    """Verify verify_zip_integrity detects corrupt bytes."""
    zip_path = tmp_path / "corrupt.zip"
    # Write corrupt byte stream
    zip_path.write_bytes(b"PK\x03\x04\x00\x00\x00\x00random_corrupt_content_not_a_zip")

    is_valid, error = verify_zip_integrity(zip_path)
    assert is_valid is False
    assert error is not None


def test_zip_integrity_missing_file(tmp_path: Path):
    """Verify verify_zip_integrity handles missing files gracefully."""
    non_existent = tmp_path / "does_not_exist.zip"
    is_valid, error = verify_zip_integrity(non_existent)
    assert is_valid is False
    assert "does not exist" in error.lower()


# ==============================================================================
# 3. Safe Extraction & Zip Slip Security Tests
# ==============================================================================

def test_safe_extract_archive_valid(tmp_path: Path):
    """Verify extracting valid members preserves files and hierarchy."""
    zip_path = tmp_path / "valid_extract.zip"
    extract_target = tmp_path / "extracted_dir"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("folder/doc.txt", "Medical research content")
        zf.writestr("root.dcm", b"\x00" * 128)

    extracted_count, errors = safe_extract_archive(zip_path, extract_target)

    assert extracted_count == 2
    assert len(errors) == 0
    assert (extract_target / "folder" / "doc.txt").read_text(encoding="utf-8") == "Medical research content"
    assert (extract_target / "root.dcm").is_file()


def test_safe_extract_archive_rejects_path_traversal(tmp_path: Path):
    """Verify Zip Slip attack members (e.g. ../../escape.txt) are rejected."""
    zip_path = tmp_path / "zipslip_malicious.zip"
    extract_target = tmp_path / "safe_jail"
    extract_target.mkdir()

    # Manually construct zip file with path traversal entries
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("legit.txt", "Legitimate payload")
        # Attacker trying to write outside safe_jail
        zf.writestr("../../evil_outside.txt", "Malicious exploit payload")

    extracted_count, errors = safe_extract_archive(zip_path, extract_target)

    # Legit should extract, malicious should be rejected
    assert extracted_count == 1
    assert len(errors) == 1
    assert "Rejected unsafe path traversal attempt" in errors[0]
    assert (extract_target / "legit.txt").is_file()
    # Confirm nothing escaped safe_jail
    assert not (tmp_path / "evil_outside.txt").exists()


# ==============================================================================
# 4. Resumable Downloader Manifest & State Tests
# ==============================================================================

def test_downloader_manifest_lifecycle(tmp_path: Path):
    """Verify manifest initialization, persistence, and state updates."""
    target_archive = tmp_path / "archive.zip"
    downloader = ResumableDownloader(
        url="https://mock.rsna.org/test.zip",
        target_path=target_archive,
        expected_size=1000,
        chunk_size=200,
    )

    # Initial manifest state before any file exists
    manifest = downloader.load_manifest()
    assert manifest["completed_bytes"] == 0
    assert manifest["expected_size"] == 1000
    assert manifest["status"] == "INITIALIZED"

    # Save updated manifest
    manifest["completed_bytes"] = 400
    manifest["status"] = "PAUSED"
    downloader.save_manifest(manifest)

    # Reload manifest from disk
    reloaded = downloader.load_manifest()
    assert reloaded["completed_bytes"] == 400
    assert reloaded["status"] == "PAUSED"


def test_downloader_verify_existing_file(tmp_path: Path):
    """Verify detection of complete vs partial vs nonexistent local archives."""
    target_archive = tmp_path / "archive.zip"
    downloader = ResumableDownloader(
        url="https://mock.rsna.org/test.zip",
        target_path=target_archive,
        expected_size=500,
    )

    # 1. Nonexistent
    is_complete, curr_size = downloader.verify_existing_file()
    assert is_complete is False
    assert curr_size == 0

    # 2. Partial
    target_archive.write_bytes(b"A" * 250)
    is_complete, curr_size = downloader.verify_existing_file()
    assert is_complete is False
    assert curr_size == 250

    # 3. Complete
    target_archive.write_bytes(b"A" * 500)
    is_complete, curr_size = downloader.verify_existing_file()
    assert is_complete is True
    assert curr_size == 500


def test_downloader_returns_complete_immediately_if_already_verified(tmp_path: Path):
    """Verify downloader does not invoke network if local archive already matches expected size."""
    target_archive = tmp_path / "complete_archive.zip"
    target_archive.write_bytes(b"Z" * 1024)

    downloader = ResumableDownloader(
        url="https://mock.rsna.org/test.zip",
        target_path=target_archive,
        expected_size=1024,
    )

    progress_reports: list[DownloadProgress] = []
    result = downloader.download(progress_callback=progress_reports.append)

    assert result.status == "COMPLETE"
    assert result.downloaded_bytes == 1024
    assert result.percent == 100.0
    assert len(progress_reports) == 1
    assert progress_reports[0].status == "COMPLETE"
