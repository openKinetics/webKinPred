# CatRange Model Files

Place the trained CatRange XGBoost model binaries here before running full
inference from the notebook.

Expected files:

```text
kcat_esmc_FINAL.pkl
km_esmc_FINAL.pkl
```

Optional standardization-stat files:

```text
kcat_esmc_FINAL_stats.pt
km_esmc_FINAL_stats.pt
```

The `.pkl` model binaries are not committed because each is hundreds of MB,
larger than GitHub's normal file-size limit.
