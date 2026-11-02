"""Focused coverage for immediate ReconXKG full-cache completion."""

from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import django
    import pandas as pd

    django.setup()
    from django.test import RequestFactory, TestCase
    from api.models import Job, JobProgressStage
    from api.services.job_service import (
        process_job_submission_from_params,
    )
    from api.services.job_progress_service import (
        increment_stage_validation,
        reset_stage_prediction_metrics,
        set_stage_prediction_progress,
        set_stage_prediction_snapshot,
        set_stage_prediction_total,
    )
    from api.services.prediction_batch_service import (
        build_sequence_batch_plan,
        build_target_batch_plan,
    )
    from api.services.recon_xkg_preflight_service import (
        ReconXkgCacheSnapshot,
        ReconXkgPreflightResult,
        preflight_recon_xkg_cache,
    )
    from api.services.result_service import serialize_result_csv
    from api.services.similarity_service import append_kcat_similarity_columns_to_output_csv
    from api.tasks import run_recon_xkg_cache_prediction
    from api.views.v1_views import api_submit_job

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    TestCase = unittest.TestCase


class _Descriptor:
    key = "PairMethod"
    display_name = "Pair method"
    max_seq_len = 4
    model_version = "1"
    col_to_kwarg = {"Substrate": "substrates"}
    target_kwargs = {"kcat": {"mode": "kcat"}, "Km": {"mode": "km"}}

    def __init__(self, behavior="expanded_pair", col_to_kwarg=None, key=None):
        self.behavior = behavior
        if col_to_kwarg is not None:
            self.col_to_kwarg = col_to_kwarg
        if key is not None:
            self.key = key

    def input_behavior(self, _target):
        return self.behavior


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class BatchPlanningTests(unittest.TestCase):
    def test_expanded_plan_preserves_positions_duplicates_and_truncation(self):
        dataframe = pd.DataFrame(
            {
                "Protein Sequence": ["ABCDE", "XYZ"],
                "Substrates": [" CCO ; O ; CCO ", "N"],
            },
            index=[10, 3],
        )
        descriptor = _Descriptor()
        sequences = build_sequence_batch_plan(dataframe, [descriptor], "truncate")
        batch = build_target_batch_plan(descriptor, "kcat", dataframe, sequences)

        self.assertEqual(sequences.valid_reaction_indices, (0, 1))
        self.assertEqual(batch.sequences, ("ABDE", "ABDE", "ABDE", "XYZ"))
        self.assertEqual(batch.call_kwargs["substrates"], ["CCO", "O", "CCO", "N"])
        self.assertEqual(batch.call_kwargs["mode"], "kcat")

    def test_native_multi_and_full_reaction_batches(self):
        dataframe = pd.DataFrame(
            {
                "Protein Sequence": ["AAA", "BBB"],
                "Substrates": ["CCO;O", "N"],
                "Products": ["CC=O;O", "N"],
            }
        )
        native_multi = _Descriptor("native_multi", key="CatPred")
        sequence_plan = build_sequence_batch_plan(dataframe, [native_multi], "skip")
        catpred = build_target_batch_plan(
            native_multi, "kcat", dataframe, sequence_plan
        )
        self.assertEqual(catpred.sequences, ("AAA", "BBB"))
        self.assertEqual(catpred.call_kwargs["substrates"], [["CCO", "O"], ["N"]])

        turnup = _Descriptor(
            "native_full_reaction",
            col_to_kwarg={"Substrates": "substrates", "Products": "products"},
            key="TurNup",
        )
        full = build_target_batch_plan(turnup, "kcat", dataframe, sequence_plan)
        self.assertEqual(full.call_kwargs["substrates"], ["CCO;O", "N"])
        self.assertEqual(full.call_kwargs["products"], ["CC=O;O", "N"])

    def test_skipped_sequences_create_no_target_units(self):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["TOOLONG", "AAA"], "Substrate": ["C", "O"]}
        )
        descriptor = _Descriptor()
        sequence_plan = build_sequence_batch_plan(dataframe, [descriptor], "skip")
        batch = build_target_batch_plan(descriptor, "Km", dataframe, sequence_plan)
        self.assertEqual(sequence_plan.valid_reaction_indices, (1,))
        self.assertEqual(batch.sequences, ("AAA",))
        self.assertEqual(batch.call_kwargs["substrates"], ["O"])

    def test_sequence_list_expands_before_substrate_list(self):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA;BBB"], "Substrates": ["C;O"]}
        )
        descriptor = _Descriptor()
        sequence_plan = build_sequence_batch_plan(dataframe, [descriptor], "truncate")
        batch = build_target_batch_plan(descriptor, "kcat", dataframe, sequence_plan)

        self.assertEqual(batch.sequences, ("AAA", "AAA", "BBB", "BBB"))
        self.assertEqual(batch.call_kwargs["substrates"], ["C", "O", "C", "O"])
        self.assertIsNotNone(batch.unit_expansion)


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA", "AAA"], "Substrates": ["C;O", "C"]}
        )
        self.descriptor = _Descriptor()

    def test_deduplicates_lookup_keys_and_captures_similarity_snapshot(self):
        key_batches = iter([(["a", "b", "a"], [None] * 3, "fp")])
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            side_effect=lambda *_a, **_k: next(key_batches),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": 2.0},
        ) as get_many, patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={"AAA": (4.0, 8.0)},
        ):
            result = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )

        self.assertTrue(result.complete)
        self.assertEqual(set(get_many.call_args.args[0]), {"a", "b"})
        self.assertEqual(result.snapshot.predictions, {"a": 1.0, "b": 2.0})
        self.assertEqual(result.snapshot.similarities, {"AAA": (4.0, 8.0)})

    def test_preflight_similarity_includes_skipped_output_rows(self):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAAAAA", "AAA"], "Substrate": ["C", "O"]}
        )
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a"], [None], "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0},
        ), patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={"AAA": (4.0, 8.0)},
        ) as get_similarity:
            result = preflight_recon_xkg_cache(
                dataframe=dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="skip",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )

        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "similarity-cache-miss")
        self.assertEqual(set(get_similarity.call_args.args[0]), {"AAAAAA", "AAA"})

    def test_preflight_similarity_uses_cached_winning_sequence(self):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA;CCC"], "Substrates": ["C"]}
        )
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b"], [None, None], "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": 2.0},
        ), patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={"CCC": (5.0, 9.0)},
        ) as get_similarity:
            result = preflight_recon_xkg_cache(
                dataframe=dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )

        self.assertTrue(result.complete)
        self.assertEqual(set(get_similarity.call_args.args[0]), {"CCC"})
        self.assertEqual(result.snapshot.similarities, {"CCC": (5.0, 9.0)})

    def test_preflight_similarity_accepts_cached_invalid_loser(self):
        from api.services.prediction_store import CachedFailure

        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AXZ;CCC;AAA"], "Substrates": ["C"]}
        )
        reason = "Invalid protein sequence (unsupported amino acid characters)"
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["bad", "winner", "loser"], [None, None, None], "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={
                "bad": CachedFailure(reason),
                "winner": 12.0,
                "loser": 0.2,
            },
        ), patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={"CCC": (5.0, 9.0)},
        ) as get_similarity:
            result = preflight_recon_xkg_cache(
                dataframe=dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )

        self.assertTrue(result.complete)
        self.assertEqual(set(get_similarity.call_args.args[0]), {"CCC"})
        self.assertEqual(result.snapshot.predictions["bad"], CachedFailure(reason))

    def test_one_missing_or_uncacheable_unit_forces_queue_path(self):
        for keys, cached, expected_reason in [
            (["a", "b", "a"], {"a": 1.0}, "prediction-cache-miss"),
            (["a", "b", "a"], {"a": 1.0, "b": float("nan")}, "prediction-cache-miss"),
            (["a", None, "a"], {}, "uncacheable-prediction-unit"),
        ]:
            with self.subTest(reason=expected_reason), patch(
                "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
                return_value=(keys, [None] * len(keys), "fp"),
            ), patch(
                "api.services.recon_xkg_preflight_service.prediction_store.get_many",
                return_value=cached,
            ):
                result = preflight_recon_xkg_cache(
                    dataframe=self.dataframe,
                    targets=["kcat"],
                    descriptors={"kcat": self.descriptor},
                    handle_long_sequences="truncate",
                    canonicalize_substrates=True,
                    include_similarity_columns=False,
                    job_public_id="job-1",
                )
            self.assertFalse(result.complete)
            self.assertEqual(result.reason, expected_reason)

    def test_negative_prediction_outcome_satisfies_preflight(self):
        from api.services.prediction_store import CachedFailure

        reason = "Invalid protein sequence (unsupported amino acid characters)"
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": CachedFailure(reason)},
        ):
            result = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=False,
                job_public_id="job-1",
            )
        self.assertTrue(result.complete)
        self.assertEqual(result.snapshot.predictions["b"], CachedFailure(reason))

    def test_missing_similarity_or_cache_exception_forces_queue_path(self):
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": 2.0},
        ), patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={},
        ):
            missing_similarity = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )
        self.assertFalse(missing_similarity.complete)
        self.assertEqual(missing_similarity.reason, "similarity-cache-miss")

        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            side_effect=RuntimeError("store unavailable"),
        ):
            unreadable = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=False,
                job_public_id="job-1",
            )
        self.assertFalse(unreadable.complete)
        self.assertEqual(unreadable.reason, "cache-read-error")


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class ReconXkgProgressScalingTests(TestCase):
    def setUp(self):
        self.job = Job.objects.create(
            public_id="job-p",
            prediction_type="kcat",
            status="Processing",
        )
        JobProgressStage.objects.create(
            job=self.job,
            stage_index=0,
            target="kcat",
            method_key="TurNup",
            method_display_name="TurNup",
            status="running",
        )

    def _stage(self):
        return JobProgressStage.objects.get(job=self.job, target="kcat")

    def test_partial_cache_progress_uses_full_denominator(self):
        set_stage_prediction_snapshot(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            molecules_total=100,
            molecules_processed=90,
            invalid_rows=0,
            predictions_total=100,
            predictions_made=90,
        )

        reset_stage_prediction_metrics(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            total_rows=10,
        )
        stage = self._stage()
        self.assertEqual(stage.predictions_total, 100)
        self.assertEqual(stage.predictions_made, 90)
        self.assertEqual(stage.molecules_total, 100)
        self.assertEqual(stage.molecules_processed, 90)

        increment_stage_validation(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            processed_inc=10,
            invalid_inc=2,
        )
        set_stage_prediction_total(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            total_predictions=8,
        )
        stage = self._stage()
        self.assertEqual(stage.predictions_total, 100)
        self.assertEqual(stage.predictions_made, 92)

        set_stage_prediction_progress(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            done=4,
            total=8,
        )
        stage = self._stage()
        self.assertEqual(stage.predictions_total, 100)
        self.assertEqual(stage.predictions_made, 96)

        set_stage_prediction_progress(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            done=8,
            total=8,
        )
        stage = self._stage()
        self.assertEqual(stage.predictions_total, 100)
        self.assertEqual(stage.predictions_made, 100)

    def test_normal_progress_is_not_scaled(self):
        reset_stage_prediction_metrics(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            total_rows=10,
        )
        set_stage_prediction_total(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            total_predictions=8,
        )
        set_stage_prediction_progress(
            job_public_id=self.job.public_id,
            target="kcat",
            method_key="TurNup",
            done=4,
            total=8,
        )
        stage = self._stage()
        self.assertEqual(stage.predictions_total, 8)
        self.assertEqual(stage.predictions_made, 4)


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class CacheTaskTests(unittest.TestCase):
    def setUp(self):
        self.job = SimpleNamespace(
            public_id="job-1",
            pk=1,
            status="Pending",
            handle_long_sequences="truncate",
        )
        self.params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "canonicalize_substrates": True,
            "include_similarity_columns": False,
            "disable_gpu_precompute": False,
        }
        self.snapshot = ReconXkgCacheSnapshot({"key": 1.0}, {})
        self.hit = ReconXkgPreflightResult(True, self.snapshot, "full-cache-hit", 1, 1, 0)
        self.miss = ReconXkgPreflightResult(
            False, None, "prediction-cache-miss", 1, 1, 0
        )

    def _run_task(self):
        run_recon_xkg_cache_prediction.run(
            self.job.public_id,
            self.params["targets"],
            self.params["methods"],
            {},
            self.params["canonicalize_substrates"],
            self.params["include_similarity_columns"],
            self.params["disable_gpu_precompute"],
        )

    def test_full_hit_runs_strict_executor_without_normal_queueing(self):
        with patch("api.tasks.Job.objects.get", return_value=self.job), patch(
            "api.tasks.get_method", return_value=_Descriptor()
        ), patch(
            "api.tasks._load_recon_xkg_preflight_dataframe", return_value=MagicMock()
        ), patch(
            "api.tasks.preflight_recon_xkg_cache", return_value=self.hit
        ), patch("api.tasks.execute_multi_prediction_job") as execute, patch(
            "api.tasks._dispatch_recon_xkg_fallback_task"
        ) as fallback:
            self._run_task()

        fallback.assert_not_called()
        execute.assert_called_once()
        self.assertTrue(execute.call_args.kwargs["cache_only"])
        self.assertTrue(execute.call_args.kwargs["recon_xkg"])
        self.assertEqual(execute.call_args.kwargs["prediction_cache_snapshot"], {"key": 1.0})

    def test_preflight_miss_dispatches_one_normal_recon_task(self):
        with patch("api.tasks.Job.objects.get", return_value=self.job), patch(
            "api.tasks.get_method", return_value=_Descriptor()
        ), patch(
            "api.tasks._load_recon_xkg_preflight_dataframe", return_value=MagicMock()
        ), patch(
            "api.tasks.preflight_recon_xkg_cache", return_value=self.miss
        ), patch("api.tasks.execute_multi_prediction_job") as execute, patch(
            "api.tasks._dispatch_recon_xkg_fallback_task"
        ) as fallback:
            self._run_task()

        execute.assert_not_called()
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["reason"], "prediction-cache-miss")

    def test_preflight_error_dispatches_one_normal_recon_task(self):
        with patch("api.tasks.Job.objects.get", return_value=self.job), patch(
            "api.tasks.get_method", return_value=_Descriptor()
        ), patch(
            "api.tasks._load_recon_xkg_preflight_dataframe", return_value=MagicMock()
        ), patch(
            "api.tasks.preflight_recon_xkg_cache", side_effect=RuntimeError("store down")
        ), patch("api.tasks.execute_multi_prediction_job") as execute, patch(
            "api.tasks._dispatch_recon_xkg_fallback_task"
        ) as fallback:
            self._run_task()

        execute.assert_not_called()
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["reason"], "preflight-error")

    def test_assembly_failure_resets_job_for_one_normal_dispatch(self):
        with patch("api.tasks.Job.objects.get", return_value=self.job), patch(
            "api.tasks.get_method", return_value=_Descriptor()
        ), patch(
            "api.tasks._load_recon_xkg_preflight_dataframe", return_value=MagicMock()
        ), patch(
            "api.tasks.preflight_recon_xkg_cache", return_value=self.hit
        ), patch(
            "api.tasks.execute_multi_prediction_job",
            side_effect=RuntimeError("assembly failed"),
        ), patch("api.tasks._reset_job_after_cache_only_failure") as reset, patch(
            "api.tasks._dispatch_recon_xkg_fallback_task"
        ) as fallback:
            self._run_task()

        reset.assert_called_once_with(self.job)
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["reason"], "cache-only-assembly-failed")


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class SubmissionDispatchTests(unittest.TestCase):
    def _submit(self, recon_xkg: bool):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA"], "Substrate": ["C"]}
        )
        params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "use_experimental": False,
            "include_similarity_columns": False,
            "canonicalize_substrates": True,
            "disable_gpu_precompute": False,
            "recon_xkg": recon_xkg,
        }
        job = SimpleNamespace(public_id="job-1")
        with patch(
            "api.services.job_service.validate_prediction_parameters", return_value=None
        ), patch(
            "api.services.job_service.validate_sequence_handling_option", return_value=None
        ), patch("api.services.job_service.parse_csv_file", return_value=dataframe), patch(
            "api.services.job_service.validate_required_columns_for_methods",
            return_value=None,
        ), patch(
            "api.services.job_service.validate_column_emptiness", return_value=None
        ), patch(
            "api.services.job_service.validate_products_column", return_value=[]
        ), patch(
            "api.services.job_service.handle_quota_validation", return_value=None
        ), patch(
            "api.services.job_service.get_experimental_results", return_value={}
        ), patch("api.services.job_service.create_job_record", return_value=job), patch(
            "api.services.job_service.create_job_directory", return_value="/tmp/job-1"
        ), patch(
            "api.services.job_service.save_job_input_file"
        ), patch(
            "api.services.job_service.dispatch_recon_xkg_cache_task"
        ) as cache_dispatch, patch(
            "api.services.job_service.dispatch_prediction_task"
        ) as dispatch:
            error, success = process_job_submission_from_params(
                params,
                MagicMock(),
                "127.0.0.1",
                user=MagicMock(),
            )
        return error, success, cache_dispatch, dispatch

    def test_recon_submission_dispatches_cache_task_only(self):
        error, success, cache_dispatch, dispatch = self._submit(True)
        self.assertIsNone(error)
        self.assertFalse(success["completed_immediately"])
        cache_dispatch.assert_called_once()
        dispatch.assert_not_called()

    def test_non_recon_submission_does_not_preflight(self):
        error, success, cache_dispatch, dispatch = self._submit(False)
        self.assertIsNone(error)
        self.assertFalse(success["completed_immediately"])
        cache_dispatch.assert_not_called()
        dispatch.assert_called_once()


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class ApiSubmitResponseTests(unittest.TestCase):
    def test_v1_submit_returns_pending_recon_job_without_result_assembly(self):
        request = RequestFactory().post("/api/v1/submit/", data={})
        request.api_key = SimpleNamespace(pk=20)
        request.api_user = MagicMock()
        request.api_request_ip = "127.0.0.1"
        request.api_quota_subject = "apikey:20"
        request.api_daily_limit = 1000

        params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "use_experimental": False,
            "include_similarity_columns": False,
            "canonicalize_substrates": True,
            "disable_gpu_precompute": False,
            "recon_xkg": True,
        }
        success = {"public_id": "job-1", "completed_immediately": False}

        with patch(
            "api.views.v1_views._parse_multipart_body",
            return_value=(MagicMock(), params, None),
        ), patch(
            "api.views.v1_views.resolve_recon_xkg", return_value=True
        ), patch(
            "api.views.v1_views.process_job_submission_from_params",
            return_value=(None, success),
        ) as submit:
            response = inspect.unwrap(api_submit_job)(request)

        submit.assert_called_once()
        self.assertEqual(response.status_code, 201)
        payload = json.loads(response.content)
        self.assertEqual(payload["jobId"], "job-1")
        self.assertEqual(payload["status"], "Pending")
        self.assertEqual(payload["statusUrl"], "/api/v1/status/job-1/")
        self.assertEqual(payload["resultUrl"], "/api/v1/result/job-1/")


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class ResultSerializationTests(unittest.TestCase):
    def test_blank_cells_are_json_null(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("prediction,source\n1.0,model\n,\n")
            output.flush()
            result = serialize_result_csv(output.name)

        self.assertEqual(result["rowCount"], 2)
        self.assertIsNone(result["data"][1]["prediction"])
        self.assertIsNone(result["data"][1]["source"])

    def test_similarity_snapshot_never_invokes_mmseqs(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAAA,1.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", None),
            ), patch("api.services.similarity_service._compute_mmseqs_similarity") as mmseqs:
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                    cached_similarity_snapshot={"AAA": (12.0, 34.0)},
                    cache_only=True,
                )
            result = serialize_result_csv(output.name)

        mmseqs.assert_not_called()
        self.assertEqual(
            result["data"][0]["mean similarity to PairMethod training data"],
            12.0,
        )
        self.assertEqual(
            result["data"][0]["max similarity to PairMethod training data"],
            34.0,
        )

    def test_negative_similarity_snapshot_renders_blank_without_mmseqs(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAAA,1.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", None),
            ), patch("api.services.similarity_service._compute_mmseqs_similarity") as mmseqs:
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                    cached_similarity_snapshot={"AAA": (None, None)},
                    cache_only=True,
                )
            result = serialize_result_csv(output.name)

        mmseqs.assert_not_called()
        self.assertIsNone(
            result["data"][0]["mean similarity to PairMethod training data"]
        )
        self.assertIsNone(
            result["data"][0]["max similarity to PairMethod training data"]
        )

    def test_invalid_similarity_sequence_is_cached_as_blank(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAXZ,1.0\nAAA,2.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", "/tmp/pairdb"),
            ), patch(
                "api.services.similarity_service.os.path.exists",
                return_value=True,
            ), patch(
                "api.services.prediction_store.get_similarity_many",
                return_value={},
            ), patch(
                "api.services.prediction_store.upsert_similarity_many",
                return_value=1,
            ) as upsert, patch(
                "api.services.similarity_service._compute_mmseqs_similarity",
                return_value=({"AAA": 34.0}, {"AAA": 12.0}),
            ) as mmseqs:
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                )
            result = serialize_result_csv(output.name)

        mmseqs.assert_called_once()
        cached_entries = [
            entry
            for call in upsert.call_args_list
            for entry in call.args[0]
        ]
        cached_by_sequence = {entry[0]: entry[2:] for entry in cached_entries}
        self.assertEqual(cached_by_sequence["AXZ"], (None, None))
        self.assertEqual(cached_by_sequence["AAA"], (12.0, 34.0))
        self.assertIsNone(
            result["data"][0]["mean similarity to PairMethod training data"]
        )
        self.assertIsNone(
            result["data"][0]["max similarity to PairMethod training data"]
        )
        self.assertEqual(
            result["data"][1]["mean similarity to PairMethod training data"],
            12.0,
        )
        self.assertEqual(
            result["data"][1]["max similarity to PairMethod training data"],
            34.0,
        )

    def test_missing_similarity_db_is_cached_as_blank(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAAA,1.0\nCCC,2.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", "/tmp/missingdb"),
            ), patch(
                "api.services.similarity_service.os.path.exists",
                return_value=False,
            ), patch(
                "api.services.prediction_store.get_similarity_many",
                return_value={},
            ), patch(
                "api.services.prediction_store.upsert_similarity_many",
                return_value=2,
            ) as upsert, patch(
                "api.services.similarity_service._compute_mmseqs_similarity",
            ) as mmseqs:
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                )
            result = serialize_result_csv(output.name)

        mmseqs.assert_not_called()
        cached_entries = upsert.call_args.args[0]
        self.assertEqual({entry[0] for entry in cached_entries}, {"AAA", "CCC"})
        self.assertTrue(all(entry[2:] == (None, None) for entry in cached_entries))
        for row in result["data"]:
            self.assertIsNone(row["mean similarity to PairMethod training data"])
            self.assertIsNone(row["max similarity to PairMethod training data"])

    def test_mmseqs_failure_is_cached_as_blank(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAAA,1.0\nCCC,2.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", "/tmp/pairdb"),
            ), patch(
                "api.services.similarity_service.os.path.exists",
                return_value=True,
            ), patch(
                "api.services.prediction_store.get_similarity_many",
                return_value={},
            ), patch(
                "api.services.prediction_store.upsert_similarity_many",
                return_value=2,
            ) as upsert, patch(
                "api.services.similarity_service._compute_mmseqs_similarity",
                side_effect=RuntimeError("mmseqs failed"),
            ):
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                )
            result = serialize_result_csv(output.name)

        cached_entries = upsert.call_args.args[0]
        self.assertEqual({entry[0] for entry in cached_entries}, {"AAA", "CCC"})
        self.assertTrue(all(entry[2:] == (None, None) for entry in cached_entries))
        for row in result["data"]:
            self.assertIsNone(row["mean similarity to PairMethod training data"])
            self.assertIsNone(row["max similarity to PairMethod training data"])


if __name__ == "__main__":
    unittest.main()
