
from sklearn.model_selection import KFold, GroupKFold

def get_folds(sequences, groups, method="kfold", n_splits=5):
    if method == "groupkfold":
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(sequences, groups=groups))
    elif method == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(sequences))
    else:
        raise ValueError(f"Unsupported fold method: {method}")