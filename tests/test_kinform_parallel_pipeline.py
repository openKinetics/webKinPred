#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.gpu_embed_service import kinform_parallel_orchestrator as kpo
from models.KinForm.code.pseq2sites import pseq2sites_stream_worker as pseq_stream


class _FakeProc:
    def __init__(self, *, rc: int, action=None, done_after_polls: int = 1):
        self._rc = rc
        self._action = action
        self._done_after_polls = done_after_polls
        self._poll_count = 0
        self._action_done = False

    def poll(self):
        self._poll_count += 1
        if self._poll_count < self._done_after_polls:
            return None
        if not self._action_done and self._action is not None:
            self._action()
            self._action_done = True
        return self._rc

    def terminate(self):
        return None

    def kill(self):
        return None

    def wait(self, timeout=None):
        return self._rc


class KinFormParallelHelpersTests(unittest.TestCase):
    def test_weighted_mean_from_residue(self):
        residue = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        weights = np.array([1.0, 3.0], dtype=np.float64)
        out = kpo.weighted_mean_from_residue(residue, weights)
        # normalized weights [0.25, 0.75]
        np.testing.assert_allclose(out, np.array([4.0, 6.0], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_weighted_mean_length_mismatch(self):
        residue = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        weights = np.array([1.0, 2.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            kpo.weighted_mean_from_residue(residue, weights)

    def test_weighted_mean_truncates_long_residue_embedding_to_binding_window(self):
        residue = np.arange(1025, dtype=np.float32).reshape(1025, 1)
        weights = np.ones(1024, dtype=np.float64)

        out = kpo.weighted_mean_from_residue(residue, weights)

        expected_window = np.concatenate([residue[:512], residue[-512:]], axis=0)
        np.testing.assert_allclose(out, expected_window.mean(axis=0), rtol=1e-5, atol=1e-5)

    def test_weighted_mean_zero_sum_falls_back_to_global_mean(self):
        residue = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        weights = np.array([0.0, 0.0], dtype=np.float64)
        out = kpo.weighted_mean_from_residue(residue, weights)
        np.testing.assert_allclose(out, residue.mean(axis=0), rtol=1e-5, atol=1e-5)

    def test_weighted_mean_non_finite_weights_fall_back_to_global_mean(self):
        residue = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        weights = np.array([np.nan, np.inf], dtype=np.float64)
        out = kpo.weighted_mean_from_residue(residue, weights)
        np.testing.assert_allclose(out, residue.mean(axis=0), rtol=1e-5, atol=1e-5)

    def test_merge_binding_site_rows_atomic_preserves_existing(self):
        with tempfile.TemporaryDirectory(prefix="pseq_stream_rows_") as tmp:
            tsv = Path(tmp) / "binding_sites_all.tsv"
            tsv.write_text("PDB\tPred_BS_Scores\nsid_a\t0.1,0.2\n", encoding="utf-8")
            pseq_stream.merge_binding_site_rows_atomic(
                tsv,
                {"sid_b": "0.3,0.4", "sid_a": "0.9,0.9"},
            )
            rows = pseq_stream._read_binding_site_rows(tsv)
            self.assertEqual(rows["sid_a"], "0.9,0.9")
            self.assertEqual(rows["sid_b"], "0.3,0.4")

    def test_pseq_stream_inputs_use_first_last_binding_window(self):
        sequence = "A" * 1025
        features = np.arange(1025, dtype=np.float32).reshape(1025, 1)

        out_sequence, out_features = pseq_stream._binding_site_model_inputs(sequence, features)

        self.assertEqual(len(out_sequence), 1024)
        expected_features = np.concatenate([features[:512], features[-512:]], axis=0)
        np.testing.assert_array_equal(out_features, expected_features)

    def test_cleanup_removes_mean_only_residue_once_satisfied(self):
        with tempfile.TemporaryDirectory(prefix="kinform_residue_cleanup_") as tmp:
            media = Path(tmp) / "media"
            seq_id = "sid_1"
            binding_sites = media / "pseq2sites" / "binding_sites_all.tsv"
            targets = kpo.ArtifactTargets(
                seq_ids=[seq_id],
                media_path=media,
                binding_sites_path=binding_sites,
                binding_site_existing_ids={seq_id},
            )
            root = "prot_t5_layer_19"
            targets.weighted_targets[(kpo._T5_FAMILY, root)] = set()

            residue = media / "sequence_info" / root / "residue_vecs" / f"{seq_id}.npy"
            mean = media / "sequence_info" / root / "mean_vecs" / f"{seq_id}.npy"
            residue.parent.mkdir(parents=True, exist_ok=True)
            mean.parent.mkdir(parents=True, exist_ok=True)
            np.save(residue, np.ones((4, 3), dtype=np.float32))
            np.save(mean, np.ones(3, dtype=np.float32))

            removed = kpo._cleanup_satisfied_residue_files(targets=targets, bs_scores={})

            self.assertEqual(removed, 1)
            self.assertFalse(residue.exists())

    def test_cleanup_removes_already_weighted_residue_once_satisfied(self):
        with tempfile.TemporaryDirectory(prefix="kinform_residue_cleanup_") as tmp:
            media = Path(tmp) / "media"
            seq_id = "sid_1"
            binding_sites = media / "pseq2sites" / "binding_sites_all.tsv"
            targets = kpo.ArtifactTargets(
                seq_ids=[seq_id],
                media_path=media,
                binding_sites_path=binding_sites,
                binding_site_existing_ids=set(),
            )
            root = "prot_t5_last"

            residue = media / "sequence_info" / root / "residue_vecs" / f"{seq_id}.npy"
            mean = media / "sequence_info" / root / "mean_vecs" / f"{seq_id}.npy"
            weighted = media / "sequence_info" / root / "weighted_vecs" / f"{seq_id}.npy"
            for path in (residue, mean, weighted):
                path.parent.mkdir(parents=True, exist_ok=True)
            np.save(residue, np.ones((4, 3), dtype=np.float32))
            np.save(mean, np.ones(3, dtype=np.float32))
            np.save(weighted, np.ones(3, dtype=np.float32))

            removed = kpo._cleanup_satisfied_residue_files(
                targets=targets,
                bs_scores={seq_id: np.ones(4, dtype=np.float32)},
            )

            self.assertEqual(removed, 1)
            self.assertFalse(residue.exists())


class KinFormParallelOrchestratorTests(unittest.TestCase):
    def test_parallel_pipeline_retries_once_then_succeeds(self):
        with tempfile.TemporaryDirectory(prefix="kinform_parallel_orch_") as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            media = root / "media"
            repo_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "models" / "KinForm" / "code" / "protein_embeddings").mkdir(parents=True, exist_ok=True)
            (repo_root / "models" / "KinForm" / "code" / "pseq2sites").mkdir(parents=True, exist_ok=True)
            seq_id_to_seq = {"sid_1": "ACDE"}
            seq_id = "sid_1"

            counters = {"esm2_launches": 0}

            def _mk_residue_mean(root_name: str):
                res_dir = media / "sequence_info" / root_name / "residue_vecs"
                mean_dir = media / "sequence_info" / root_name / "mean_vecs"
                res_dir.mkdir(parents=True, exist_ok=True)
                mean_dir.mkdir(parents=True, exist_ok=True)
                residue = np.ones((4, 3), dtype=np.float32)
                np.save(res_dir / f"{seq_id}.npy", residue)
                np.save(mean_dir / f"{seq_id}.npy", residue.mean(axis=0))

            def _write_bs():
                bs = media / "pseq2sites" / "binding_sites_all.tsv"
                bs.parent.mkdir(parents=True, exist_ok=True)
                bs.write_text("PDB\tPred_BS_Scores\nsid_1\t1.0,1.0,1.0,1.0\n", encoding="utf-8")

            def _fake_start_worker(cmd, env):
                script = cmd[1] if len(cmd) > 1 else ""
                if script.endswith("t5_embeddings.py"):
                    return _FakeProc(rc=0, action=lambda: (_mk_residue_mean("prot_t5_layer_19"), _mk_residue_mean("prot_t5_last")))
                if script.endswith("prot_embeddings.py") and "esm2" in cmd:
                    counters["esm2_launches"] += 1
                    if counters["esm2_launches"] == 1:
                        return _FakeProc(rc=1, action=None)
                    return _FakeProc(rc=0, action=lambda: (_mk_residue_mean("esm2_layer_26"), _mk_residue_mean("esm2_layer_29")))
                if script.endswith("prot_embeddings.py") and "esmc" in cmd:
                    return _FakeProc(rc=0, action=lambda: (_mk_residue_mean("esmc_layer_24"), _mk_residue_mean("esmc_layer_32")))
                if script.endswith("pseq2sites_stream_worker.py"):
                    return _FakeProc(rc=0, action=_write_bs)
                raise AssertionError(f"Unexpected command: {cmd}")

            env = {
                "KINFORM_T5_PATH": "/tmp/fake_t5_python",
                "KINFORM_ESM_PATH": "/tmp/fake_esm_python",
                "KINFORM_ESMC_PATH": "/tmp/fake_esmc_python",
                "KINFORM_PSEQ2SITES_PATH": "/tmp/fake_pseq_python",
                "KINFORM_MEDIA_PATH": str(media),
                "GPU_REPO_ROOT": str(repo_root),
                "KINFORM_PARALLEL_LOG_LEVEL": "quiet",
            }

            with patch.object(kpo, "_start_worker", side_effect=_fake_start_worker):
                with patch.object(kpo.time, "sleep", return_value=None):
                    kpo.run_kinform_parallel_pipeline(
                        env=env,
                        repo_root=repo_root,
                        media_path=media,
                        seq_id_to_seq=seq_id_to_seq,
                        job_id="job_test",
                    )

            self.assertEqual(counters["esm2_launches"], 2)
            for root_name in (
                "prot_t5_layer_19",
                "prot_t5_last",
                "esm2_layer_26",
                "esm2_layer_29",
                "esmc_layer_24",
                "esmc_layer_32",
            ):
                weighted = media / "sequence_info" / root_name / "weighted_vecs" / f"{seq_id}.npy"
                self.assertTrue(weighted.exists(), f"missing weighted output: {weighted}")
                residue = media / "sequence_info" / root_name / "residue_vecs" / f"{seq_id}.npy"
                self.assertFalse(residue.exists(), f"residue file should be cleaned: {residue}")

    def test_stream_mode_falls_back_to_file_polling_when_torch_unavailable(self):
        env = {
            "KINFORM_PARALLEL_STREAM_ENABLE": "1",
            "KINFORM_PARALLEL_STREAM_ALLOW_LEGACY_FALLBACK": "1",
        }
        called = {"legacy": 0}

        def _fake_legacy(**kwargs):
            called["legacy"] += 1

        with patch.object(kpo, "torch", None):
            with patch.object(kpo, "_run_kinform_parallel_pipeline_file_polling", side_effect=_fake_legacy):
                kpo.run_kinform_parallel_pipeline(
                    env=env,
                    repo_root=Path("/tmp"),
                    media_path=Path("/tmp"),
                    seq_id_to_seq={"sid_1": "ACDE"},
                    job_id="job_stream_fallback",
                )

        self.assertEqual(called["legacy"], 1)


if __name__ == "__main__":
    unittest.main()
