from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import math, random

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def train_model(
    X_train, y_train, X_test, y_test,
    fold=None, sample_weight=None, n_jobs=-1, et_params=None,
    return_one_pred=True
):
    if fold is None:
        fold = random.randint(0, 10000)

    if et_params is not None:
        model = ExtraTreesRegressor(
            n_jobs=n_jobs,
            random_state=fold,
            **et_params,
        )
    else:
        model = ExtraTreesRegressor(
            n_jobs=n_jobs,
            max_features=1.0,
            random_state=fold,
        )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Model’s aggregate prediction (mean across trees)
    y_pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, y_pred), "rmse": rmse(y_test, y_pred)}

    if return_one_pred:
        return model, y_pred, metrics

    # if return_one_pred is False --> return matrix of all trees' predictions (n_trees, n_targets)
    y_pred_matrix = np.stack([est.predict(X_test) for est in model.estimators_], axis=0)
    
    return model, y_pred, metrics, y_pred_matrix
