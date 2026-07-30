"""Standalone benchmark constants copied from CatRange config/evaluation logic."""

PARAMETER_BIN_EDGES = {
    "kcat": [0, 1.0e-8, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e8],
    "km": [1.0e-14, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e4],
}

PARAMETER_UNITS = {
    "kcat": "s^(-1)",
    "km": "mM",
    "ki": "mM",
}

CATPRED_LOG_COLUMNS = {
    "kcat": "log10kcat_max",
    "km": "log10km_mean",
    "ki": "log10ki_mean",
}

CATPRED_LINEAR_COLUMNS = {
    "kcat": "Prediction_(s^(-1))",
    "km": "Prediction_(mM)",
    "ki": "Prediction_(mM)",
}

DLKCAT_OUTPUT_COLUMN = "Kcat value (1/s)"

