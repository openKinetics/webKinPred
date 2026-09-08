#!/usr/bin/env python3
"""Unit tests for shared embedding cache/staging I/O helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.gpu_embed_service.cache_io import (
    merge_manifest_entries,
    remove_manifest_entries,
    resolve_missing_ids,
)


class CacheIoTests(unittest.TestCase):
    def test_ephemeral_cleanup_removes_manifest_entry_without_per_file_stat(self):
        with tempfile.TemporaryDirectory(prefix="cache_io_manifest_") as tmp:
            cache_dir = Path(tmp)
            merge_manifest_entries(
                cache_dir,
                {
                    "sid_1": {
                        "filename": "sid_1.pt",
                        "bytes": 123,
                        "updated_at": 1.0,
                        "ready": True,
                    }
                },
            )

            # Manifest hits are trusted for speed on mounted cache dirs; cleanup
            # must remove ephemeral entries instead of forcing per-ID stat calls.
            missing, ready = resolve_missing_ids(["sid_1"], cache_dir=cache_dir, suffix=".pt")
            self.assertEqual(missing, [])
            self.assertEqual(ready, {"sid_1"})

            remove_manifest_entries(cache_dir, ["sid_1"])
            missing, ready = resolve_missing_ids(["sid_1"], cache_dir=cache_dir, suffix=".pt")
            self.assertEqual(missing, ["sid_1"])
            self.assertEqual(ready, set())

    def test_on_disk_file_missing_from_manifest_is_resolved_via_probe(self):
        # Files written outside the GPU service (CPU fallbacks / legacy paths)
        # land on disk without a manifest entry. They must still resolve as
        # ready via a targeted probe of the exact path.
        with tempfile.TemporaryDirectory(prefix="cache_io_probe_") as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "sid_present.npy").write_bytes(b"x")

            missing, ready = resolve_missing_ids(
                ["sid_present", "sid_absent"], cache_dir=cache_dir, suffix=".npy"
            )
            self.assertEqual(missing, ["sid_absent"])
            self.assertEqual(ready, {"sid_present"})

    def test_resolution_ignores_unrelated_entries_and_suffix(self):
        with tempfile.TemporaryDirectory(prefix="cache_io_probe_") as tmp:
            cache_dir = Path(tmp)
            # Right stem, wrong suffix must not count as ready.
            (cache_dir / "sid_1.pt").write_bytes(b"x")

            missing, ready = resolve_missing_ids(
                ["sid_1"], cache_dir=cache_dir, suffix=".npy"
            )
            self.assertEqual(missing, ["sid_1"])
            self.assertEqual(ready, set())


if __name__ == "__main__":
    unittest.main()
