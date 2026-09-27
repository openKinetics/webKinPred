# CatRange Model Files

Place the trained CatRange XGBoost model binaries here before running full
inference from the notebook.

Expected files:

```text
kcat_model_v1b.pkl
km_model_v1b.pkl
```

These released weights use ESM-C protein features plus ChemBERTa substrate
features (1920 total). They take precedence across all supported directories.
The documented `kcat_esmc_FINAL.pkl` and `km_esmc_FINAL.pkl` names remain
supported when the v1b files are absent.

The archive also contains `kcat_model.pkl` and `km_model.pkl`. Those belong to
the ESM-2 pathway (2048 features), with different preprocessing and range
definitions. They are not accepted as fallbacks for this ESM-C integration.

Required standardization-stat files (must match the selected model weights):

```text
kcat_esmc_FINAL_stats.pt
km_esmc_FINAL_stats.pt
```

Inference rejects missing or invalid statistics. Both the protein and substrate
embeddings must be standardized using the training statistics before prediction.
Archives may place the weights in a `model_weights/` subdirectory; keep the
statistics here, or beside the weights in that subdirectory.

OKP keeps its existing `CatRange` configuration pointing at `models/CatRange`.
Inference resolves both this repository-root layout and an explicit artifact
directory. Weights downloaded earlier into `models/CatRange/model_weights/`
reuse the tracked normalization files here. The registered model remains
`CatRange` for both kcat and Km.

The `.pkl` model binaries are not committed because each is hundreds of MB,
larger than GitHub's normal file-size limit.
