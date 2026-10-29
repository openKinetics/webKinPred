import pytest

from tools.executor.compare_cpu import compare_results


def test_xgboost_thread_limit_preserves_saved_training_parameters(monkeypatch):
    from tools.executor.policy import limit_estimator

    class FittedClassifier:
        __module__ = "xgboost.sklearn"
        n_jobs = -1
        eval_metric = ["mlogloss"]

        def get_booster(self):
            return self

        def set_param(self, values):
            assert values == {"nthread": 2}
            self.threads = values["nthread"]

        def set_params(self, **values):
            pytest.fail("Replaying the saved training parameters changes this model")

    monkeypatch.setenv("WEBKINPRED_EXECUTION_POLICY", "gpu-only")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    model = FittedClassifier()
    assert limit_estimator(model) is model
    assert model.n_jobs == model.threads == 2
    assert model.eval_metric == ["mlogloss"]


def test_cpu_parity_preserves_categories_and_order():
    baseline = {"preds": [1.2, None, "range-low"], "sources": ["predicted", "invalid", "range"]}
    compare_results(baseline, {**baseline, "preds": [1.2000001, None, "range-low"]})
    with pytest.raises(ValueError, match="categorical"):
        compare_results(baseline, {**baseline, "preds": [1.2, 0, "range-low"]})
    with pytest.raises(ValueError, match="metadata"):
        compare_results(baseline, {**baseline, "sources": list(reversed(baseline["sources"]))})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.4])
def test_cpu_parity_rejects_nonfinite_and_changed_values(value):
    with pytest.raises(ValueError, match="numerical"):
        compare_results({"preds": [1.2]}, {"preds": [value]})


def test_cpu_parity_compares_json_numeric_annotations_but_preserves_labels():
    baseline = {
        "preds": ["[0.0022782160138459]"],
        "extra": ['[{"sequenceIndex":1,"prediction":0.0022782160138459}]'],
    }
    compare_results(
        baseline,
        {
            "preds": ["[0.0022782158148355]"],
            "extra": ['[{"sequenceIndex":1,"prediction":0.0022782158148355}]'],
        },
    )
    with pytest.raises(ValueError, match="categorical"):
        compare_results(
            baseline,
            {**baseline, "extra": ['[{"sequenceIndex":2,"prediction":0.0022782160138459}]']},
        )
