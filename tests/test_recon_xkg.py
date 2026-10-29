"""
Unit tests for the ReconXKG memoization layer.

These cover the parts that do not need a live database:
  * canonicalization / lookup-key derivation (order-independence, products,
    version/param partitioning, unparseable handling),
  * the cache wrapper around the prediction engine (hit/miss partitioning,
    ordering, write-back, partial overlap, reordered substrate lists),

using an in-memory fake store and a spy engine. DB-backed I/O (get_many /
upsert_many over SQLite) is exercised by the integration suite.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import django  # noqa: E402

django.setup()

from api.services import prediction_store as store  # noqa: E402
from api.utils.recon_xkg import coerce_recon_xkg  # noqa: E402

# ---------------------------------------------------------------------------
# Keying / canonicalization
# ---------------------------------------------------------------------------


class CanonicalUnitTests(unittest.TestCase):
    def test_order_independent_and_deduplicated(self):
        a = store.canonical_unit("CCO;O", canonicalize=True)
        b = store.canonical_unit("O;CCO", canonicalize=True)
        self.assertIsNotNone(a)
        self.assertEqual(a, b, "substrate order must not change the canonical unit")

    def test_list_and_string_inputs_agree(self):
        as_str = store.canonical_unit("CCO;O", canonicalize=True)
        as_list = store.canonical_unit(["O", "CCO"], canonicalize=True)
        self.assertEqual(as_str, as_list)

    def test_canonicalizes_equivalent_smiles(self):
        # Both are ethanol; with canonicalization they collapse to one key.
        self.assertEqual(
            store.canonical_unit("OCC", canonicalize=True),
            store.canonical_unit("CCO", canonicalize=True),
        )

    def test_raw_when_not_canonicalizing(self):
        # Without canonicalization the raw (validated) text is kept, so two
        # different spellings of ethanol must NOT collide.
        self.assertNotEqual(
            store.canonical_unit("OCC", canonicalize=False),
            store.canonical_unit("CCO", canonicalize=False),
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(store.canonical_unit("not-a-molecule", canonicalize=True))
        self.assertIsNone(store.canonical_unit("", canonicalize=True))


class LookupKeyTests(unittest.TestCase):
    def _key(self, **overrides):
        base = dict(
            target="kcat",
            method="DLKcat",
            model_version="1",
            params_fp="fp",
            sequence_sha256="seqsha",
            substrate_canon="CCO",
            products_canon="",
        )
        base.update(overrides)
        return store.make_lookup_key(**base)

    def test_deterministic(self):
        self.assertEqual(self._key(), self._key())

    def test_partitions_on_every_field(self):
        baseline = self._key()
        for field, value in [
            ("target", "Km"),
            ("method", "TurNup"),
            ("model_version", "2"),
            ("params_fp", "other"),
            ("sequence_sha256", "other"),
            ("substrate_canon", "CCC"),
            ("products_canon", "O"),
        ]:
            self.assertNotEqual(baseline, self._key(**{field: value}), field)

    def test_params_fingerprint_tracks_flags(self):
        self.assertNotEqual(
            store.params_fingerprint(True, {}),
            store.params_fingerprint(False, {}),
        )
        self.assertNotEqual(
            store.params_fingerprint(True, {"kinetics_type": "KCAT"}),
            store.params_fingerprint(True, {"kinetics_type": "KM"}),
        )

    def test_invalid_sequence_receives_a_cache_key_but_empty_sequence_does_not(self):
        desc = _FakeDesc(col_to_kwarg={"Substrate": "substrates"})
        keys, components, _fingerprint = store.build_unit_keys(
            desc,
            "kcat",
            ["MAAA", "INVALIDX", ""],
            {"substrates": ["CCO", "CCO", "CCO"]},
            True,
        )
        self.assertIsNotNone(keys[0])
        self.assertIsNotNone(keys[1])
        self.assertIsNotNone(components[1])
        self.assertIsNone(keys[2])
        self.assertIsNone(components[2])

    def test_invalid_chemistry_uses_a_stable_non_reversible_fallback(self):
        desc = _FakeDesc(col_to_kwarg={"Substrate": "substrates"})
        first = store.build_unit_keys(
            desc,
            "kcat",
            ["MAAA"],
            {"substrates": ["not-a-molecule"]},
            True,
        )
        second = store.build_unit_keys(
            desc,
            "kcat",
            ["MAAA"],
            {"substrates": ["not-a-molecule"]},
            True,
        )
        self.assertEqual(first[0], second[0])
        self.assertTrue(first[1][0][1].startswith("raw-sha256:"))
        self.assertNotIn("not-a-molecule", first[1][0][1])


class CoerceTests(unittest.TestCase):
    def test_coerce_value(self):
        self.assertEqual(store.coerce_value(1.5), 1.5)
        self.assertEqual(store.coerce_value("2.0"), 2.0)
        self.assertEqual(store.coerce_value(3), 3.0)
        for bad in (None, True, "nan", "inf", "", "x", float("inf")):
            self.assertIsNone(store.coerce_value(bad), bad)

    def test_coerce_recon_xkg(self):
        for truthy in ("true", "1", "yes", "on", "TRUE", True, 1):
            self.assertTrue(coerce_recon_xkg(truthy), truthy)
        for falsy in ("false", "0", "no", "", None, "anything"):
            self.assertFalse(coerce_recon_xkg(falsy), falsy)

    def test_empty_prediction_outcomes_are_negative_cacheable(self):
        for reason in (
            "Invalid protein sequence (unsupported amino acid characters)",
            "Invalid product (not a valid SMILES or InChI)",
            "Missing substrate",
            "Unsupported substrate for CatPred: [H+] contains no heavy atoms",
            "Prediction could not be made",
            "Prediction output missing",
        ):
            self.assertTrue(store.is_cacheable_failure_reason(reason), reason)
        self.assertFalse(store.is_cacheable_failure_reason(""))
        self.assertFalse(store.is_cacheable_failure_reason(None))


# ---------------------------------------------------------------------------
# Cache wrapper (fake store + spy engine)
# ---------------------------------------------------------------------------


class _FakeDesc:
    def __init__(self, col_to_kwarg, target_kwargs=None, key="FakeKcat", model_version="1"):
        self.col_to_kwarg = col_to_kwarg
        self.target_kwargs = target_kwargs or {}
        self.key = key
        self.model_version = model_version


class CacheWrapperTests(unittest.TestCase):
    def setUp(self):
        import api.tasks as tasks

        self.tasks = tasks
        self._mem: dict[str, object] = {}
        self.engine_calls: list[list[str]] = []

        # In-memory fake store backing get_many / upsert_many.
        def fake_get_many(keys):
            return {k: self._mem[k] for k in set(keys) if k in self._mem}

        def fake_upsert_many(rows):
            for row in rows:
                reason = row.get("failure_reason")
                self._mem[row["lookup_key"]] = (
                    store.CachedFailure(reason) if reason else row["value"]
                )
            return len(rows)

        # Spy engine: returns a deterministic value per substrate and records
        # which substrates it was asked to compute.
        def fake_engine(desc, sequences, public_id, target, **kwargs):
            substrates = kwargs.get("substrates", [])
            self.engine_calls.append(list(substrates))
            return [float(len(str(s))) for s in substrates], {}, None

        self._patches = [
            ("api.services.prediction_store", "get_many", fake_get_many),
            ("api.services.prediction_store", "upsert_many", fake_upsert_many),
        ]
        import api.services.prediction_store as ps

        self._orig = {}
        self._orig[(ps, "get_many")] = ps.get_many
        self._orig[(ps, "upsert_many")] = ps.upsert_many
        ps.get_many = fake_get_many
        ps.upsert_many = fake_upsert_many

        self._orig[(tasks, "_run_method_engine")] = tasks._run_method_engine
        self._orig[(tasks, "set_stage_prediction_snapshot")] = tasks.set_stage_prediction_snapshot
        tasks._run_method_engine = fake_engine
        tasks.set_stage_prediction_snapshot = lambda **_kw: None

    def tearDown(self):
        for (mod, name), value in self._orig.items():
            setattr(mod, name, value)

    def _invoke(self, substrates, sequences=None, target="kcat"):
        desc = _FakeDesc(col_to_kwarg={"Substrate": "substrates"}, target_kwargs={"kcat": {}})
        sequences = sequences or ["MAAA"] * len(substrates)
        stats = {"hits": 0, "misses": 0, "units": 0}
        preds, invalid, _extra = self.tasks._invoke_method_prediction(
            desc,
            sequences,
            "job1",
            target,
            canonicalize_substrates=True,
            disable_gpu_precompute=False,
            recon_xkg=True,
            cache_stats=stats,
            substrates=substrates,
        )
        return preds, invalid, stats

    def test_first_run_all_miss_then_full_hit(self):
        preds1, _inv, stats1 = self._invoke(["CCO", "CCC"])
        self.assertEqual(len(self.engine_calls), 1)
        self.assertEqual(stats1["misses"], 2)
        self.assertEqual(stats1["hits"], 0)

        self.engine_calls.clear()
        preds2, _inv2, stats2 = self._invoke(["CCO", "CCC"])
        self.assertEqual(self.engine_calls, [], "fully cached run must not call the engine")
        self.assertEqual(stats2["hits"], 2)
        self.assertEqual(preds1, preds2, "cached values must match freshly computed ones")

    def test_partial_overlap_only_computes_new(self):
        self._invoke(["CCO", "CCC"])
        self.engine_calls.clear()
        preds, _inv, stats = self._invoke(["CCO", "CCCC"])
        # Only the unseen substrate is computed.
        self.assertEqual(self.engine_calls, [["CCCC"]])
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(len(preds), 2)

    def test_reordered_substrates_hit_same_entries(self):
        self._invoke(["CCO", "CCC"])
        self.engine_calls.clear()
        # Same molecules, different row order — still all hits (per-unit keying).
        _preds, _inv, stats = self._invoke(["CCC", "CCO"])
        self.assertEqual(self.engine_calls, [])
        self.assertEqual(stats["hits"], 2)

    def test_results_returned_in_input_order(self):
        # Prime the cache for one substrate, then submit a mixed batch and
        # confirm positions are preserved across the hit/miss merge.
        self._invoke(["CCO"])
        preds, _inv, _stats = self._invoke(["CCC", "CCO", "CCCC"])
        # Spy returns len(substrate); cached CCO == 3.0 must land at index 1.
        self.assertEqual(preds[1], 3.0)
        self.assertEqual(len(preds), 3)

    def test_deterministic_validation_failure_is_cached_and_replayed(self):
        reason = "Invalid protein sequence (unsupported amino acid characters)"

        def invalid_engine(desc, sequences, public_id, target, **kwargs):
            self.engine_calls.append(list(sequences))
            return [None] * len(sequences), {index: reason for index in range(len(sequences))}, None

        self.tasks._run_method_engine = invalid_engine
        predictions, invalid, stats = self._invoke(["CCO"], sequences=["MAAX"])
        self.assertEqual(predictions, [None])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(len(self.engine_calls), 1)

        self.engine_calls.clear()
        predictions, invalid, stats = self._invoke(["CCO"], sequences=["MAAX"])
        self.assertEqual(predictions, [None])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(self.engine_calls, [])

    def test_generic_row_failure_is_cached_and_replayed(self):
        reason = "Prediction could not be made"

        def failed_engine(desc, sequences, public_id, target, **kwargs):
            self.engine_calls.append(list(sequences))
            return [None] * len(sequences), {0: reason}, None

        self.tasks._run_method_engine = failed_engine
        _predictions, invalid, _stats = self._invoke(["CCO"])
        self.assertEqual(invalid, {0: reason})
        self.engine_calls.clear()

        _predictions, invalid, stats = self._invoke(["CCO"])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(self.engine_calls, [])

    def test_missing_prediction_value_is_cached_and_replayed_as_blank(self):
        reason = "Prediction could not be made"

        def blank_engine(desc, sequences, public_id, target, **kwargs):
            self.engine_calls.append(list(sequences))
            return [None] * len(sequences), {}, None

        self.tasks._run_method_engine = blank_engine
        predictions, invalid, stats = self._invoke(["CCO"])
        self.assertEqual(predictions, [None])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(len(self.engine_calls), 1)

        self.engine_calls.clear()
        predictions, invalid, stats = self._invoke(["CCO"])
        self.assertEqual(predictions, [None])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(self.engine_calls, [])

    def test_cache_only_snapshot_never_reads_store_or_invokes_engine(self):
        desc = _FakeDesc(
            col_to_kwarg={"Substrate": "substrates"},
            target_kwargs={"kcat": {}},
        )
        keys, _components, _fingerprint = self.tasks._recon_xkg_unit_keys(
            desc,
            "kcat",
            ["MAAA", "MAAA"],
            {"substrates": ["CCO", "O"]},
            True,
        )
        snapshot = {keys[0]: 3.0, keys[1]: 1.0}

        import api.services.prediction_store as prediction_store

        original_get_many = prediction_store.get_many
        prediction_store.get_many = lambda _keys: self.fail("cache-only mode read the store")
        self.engine_calls.clear()
        try:
            predictions, invalid, _extra = self.tasks._invoke_method_prediction(
                desc,
                ["MAAA", "MAAA"],
                "job1",
                "kcat",
                canonicalize_substrates=True,
                recon_xkg=True,
                cache_snapshot=snapshot,
                cache_only=True,
                substrates=["CCO", "O"],
            )
        finally:
            prediction_store.get_many = original_get_many

        self.assertEqual(predictions, [3.0, 1.0])
        self.assertEqual(invalid, {})
        self.assertEqual(self.engine_calls, [])

    def test_cache_only_snapshot_missing_value_raises_without_engine(self):
        desc = _FakeDesc(
            col_to_kwarg={"Substrate": "substrates"},
            target_kwargs={"kcat": {}},
        )
        self.engine_calls.clear()
        with self.assertRaises(self.tasks.ReconXkgCacheOnlyMiss):
            self.tasks._invoke_method_prediction(
                desc,
                ["MAAA"],
                "job1",
                "kcat",
                canonicalize_substrates=True,
                recon_xkg=True,
                cache_snapshot={},
                cache_only=True,
                substrates=["CCO"],
            )
        self.assertEqual(self.engine_calls, [])

    def test_cache_only_snapshot_replays_negative_hit_without_engine(self):
        reason = "Invalid substrate (not a valid SMILES or InChI)"
        desc = _FakeDesc(
            col_to_kwarg={"Substrate": "substrates"},
            target_kwargs={"kcat": {}},
        )
        keys, _components, _fingerprint = self.tasks._recon_xkg_unit_keys(
            desc,
            "kcat",
            ["MAAA"],
            {"substrates": ["not-a-molecule"]},
            True,
        )
        self.engine_calls.clear()
        predictions, invalid, _extra = self.tasks._invoke_method_prediction(
            desc,
            ["MAAA"],
            "job1",
            "kcat",
            canonicalize_substrates=True,
            recon_xkg=True,
            cache_snapshot={keys[0]: store.CachedFailure(reason)},
            cache_only=True,
            substrates=["not-a-molecule"],
        )
        self.assertEqual(predictions, [None])
        self.assertEqual(invalid, {0: reason})
        self.assertEqual(self.engine_calls, [])


if __name__ == "__main__":
    unittest.main()
