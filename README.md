<div align="center">

# Open Kinetics Predictor

**Predict enzyme kinetic parameters (<i>k<sub>cat</sub></i> and <i>K<sub>M</sub></i>) from protein sequence and substrate SMILES.**

[![Live Service](https://img.shields.io/badge/live-predictor.openkinetics.org-success)](https://predictor.openkinetics.org/)
[![License](https://img.shields.io/github/license/openKinetics/webKinPred)](LICENSE)
[![Stars](https://img.shields.io/github/stars/openKinetics/webKinPred?style=social)](https://github.com/your-org/open-kinetics-predictor/stargazers)
[![Forks](https://img.shields.io/github/forks/openKinetics/webKinPred?style=social)](https://github.com/your-org/open-kinetics-predictor/network/members)
[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

[Live Demo](https://predictor.openkinetics.org/) ·
[API Docs](https://predictor.openkinetics.org/api-docs) ·
[Contributing](docs/project/contributing.rst)

</div>

Open Kinetics Predictor is a production web interface for predicting enzyme kinetic parameters (kcat and KM) from protein sequence and substrate SMILES. It consolidates several state‑of‑the‑art machine learning / deep learning models behind a unified, asynchronous job API so you can submit sequences and retrieve structured predictions.

**Live service:** [https://predictor.openkinetics.org/](https://predictor.openkinetics.org/)

## Prediction Engines

| Engine | Input needed | Output | Citation |
|--------|--------------|--------|----------|
| KinForm-H | Protein sequence + substrate SMILES | kcat or Km | [Alwer & Fleming, npj Syst Biol Appl 2026](https://www.nature.com/articles/s41540-026-00692-5) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| KinForm-L | Protein sequence + substrate SMILES | kcat | [Alwer & Fleming, npj Syst Biol Appl 2026](https://www.nature.com/articles/s41540-026-00692-5) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| UniKP | Protein sequence + substrate SMILES | kcat or Km | [Yu et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-44113-1) ([GitHub](https://github.com/Luo-SynBioLab/UniKP)) |
| DLKcat | Protein sequence + substrate SMILES | kcat | [Li et al., Nat Catal 2022](https://www.nature.com/articles/s41929-022-00798-z) ([GitHub](https://github.com/SysBioChalmers/DLKcat)) |
| TurNup | Protein sequence + substrates list + products list | kcat | [Kroll et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-39840-4) ([GitHub](https://github.com/AlexanderKroll/Kcat_prediction)) |
| EITLEM | Protein sequence + substrate SMILES | kcat or Km | [Shen et al., Biotechnol Adv 2024](https://www.sciencedirect.com/science/article/pii/S2667109324002665) ([GitHub](https://github.com/XvesS/EITLEM-Kinetics)) |
| CataPro | Protein sequence + substrate SMILES | kcat, Km, or kcat/Km | [Wang et al., Nat Commun 2025](https://www.nature.com/articles/s41467-025-58038-4) ([GitHub](https://github.com/zchwang/CataPro)) |
| CatPred | Protein sequence + substrate SMILES | kcat or Km | [Boorla et al., Nat Commun 2025](https://www.nature.com/articles/s41467-025-57215-9) ([GitHub](https://github.com/maranasgroup/CatPred)) |
| OmniESI | Protein sequence + substrate SMILES | kcat or Km | [Nie et al., arXiv 2025](https://doi.org/10.48550/arXiv.2506.17963) ([GitHub](https://github.com/Hong-yu-Zhang/OmniESI)) |
| CatRange | Protein sequence + substrate SMILES | kcat or Km | [Sajeevan et al., bioRxiv 2025](https://doi.org/10.1101/2025.02.10.637555) ([GitHub](https://github.com/ssbio/CatRange)) |
| IECata | Protein sequence + substrate SMILES | kcat/Km | [Wang et al., Brief Bioinform 2025](https://doi.org/10.1093/bib/bbaf283) ([GitHub](https://github.com/zhaoyanpeng208/IECata)) |
| MMISA-KM | Protein sequence + substrate SMILES | Km | [Song & Wang, DDCLS 2025](https://doi.org/10.1109/DDCLS66240.2025.11064981) ([GitHub](https://github.com/kaiwang-group/MMISA-KM)) |

Each model is loaded with its published weights/code from `models/` and invoked through integration wrappers in `api/prediction_engines/`, so new engines can be added with minimal wiring.

## Adding a New Prediction Method

See [docs/project/contributing.rst](docs/project/contributing.rst) for a step-by-step guide.

## Features

* Batch submission of sequences and substrates.
* Long‑running inference handled asynchronously (Celery + Redis) with progress tracking.
* Sequence similarity distribution of input data vs mehtods' training data (Using mmseq2).
* Caching sequence embeddings.

## Stack

### Frontend

* React 18 + Vite (fast dev + ESM build)
* Bootstrap / React‑Bootstrap for layout & components
* Axios for API calls; Chart.js for result visualisation

### Backend

* Django 5.1 (REST-style endpoints under `api/`)
* Celery workers for queued prediction tasks (`api/tasks.py`)
* Redis as Celery broker
* SQLite
* PyTorch, scikit-learn, RDKit, pandas for model computation & cheminformatics

## Required Environment Variable

`DJANGO_SECRET_KEY` is required at runtime (no fallback hardcoded key).

Generate a strong key:

```bash
openssl rand -hex 50
```

Local/dev setup:

```bash
cp .env.example .env
# edit .env and set DJANGO_SECRET_KEY=...
docker compose up -d --build
```

Production setup:

```bash
# on the production host
cp .env.example .env.production
# edit .env.production and set DJANGO_SECRET_KEY=...
./deploy.sh prod
```

`deploy.sh prod` now reads `.env.production` automatically (falls back to `.env` if present).
For non-interactive deploys (for example CI/systemd), provide an env file or inject
`DJANGO_SECRET_KEY` in the service environment before running `docker compose`.

## High-Level Flow

1. User submits a job (sequence + substrate(s) [+ products/mutant context if required]) via the frontend.
2. Backend validates input (`api/services/validation_service.py`).
3. A Celery task is enqueued; Redis broker stores the task message.
4. Worker loads the selected model wrapper (e.g. `prediction_engines/kinform.py`) and executes inference.
5. Results & intermediate status are persisted; cached for repeated identical queries.
6. Frontend polls job status endpoint to update progress and results.

## API Access

OpenKineticsPredictor provides a REST API for programmatic access. Submit prediction
jobs, poll their status, and download results — no web browser required.

**Base URL:** `https://predictor.openkinetics.org/api/v1`

> **Full interactive documentation** is also available on the live site at
> [`/api-docs`](https://predictor.openkinetics.org/api-docs).

---

### Endpoint Overview

| Method | Endpoint | Auth | Description |
| ------ | -------- | ---- | ----------- |
| `GET` | `/health/` | No | Service health check |
| `GET` | `/methods/` | No | List available methods and required columns |
| `GET` | `/quota/` | Yes | Check remaining daily quota |
| `POST` | `/validate/` | Yes | Validate input data without submitting a job |
| `POST` | `/submit/` | Yes | Submit a prediction job |
| `GET` | `/status/<jobId>/` | Yes | Poll job status and progress |
| `GET` | `/result/<jobId>/` | Yes | Download results (CSV or `?format=json`) |

---

### Quick Start — Python

```python
import requests
import time

API_KEY = "ak_your_key_here"
BASE    = "https://predictor.openkinetics.org/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 1. Submit a job
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/submit/",
        headers=HEADERS,
        files={"file": f},
        data={
            "predictionType":      "kcat",
            "kcatMethod":          "DLKcat",
            "handleLongSequences": "truncate",
            "useExperimental":     "true",
        },
    )
resp.raise_for_status()
job = resp.json()
print(f"Job ID: {job['jobId']}  |  Quota remaining: {job['quota']['remaining']:,}")

# 2. Poll until complete
while True:
    status = requests.get(f"{BASE}/status/{job['jobId']}/", headers=HEADERS).json()
    print(f"  {status['status']} ({status['elapsedSeconds']}s)")
    if status["status"] == "Completed":
        break
    if status["status"] == "Failed":
        raise RuntimeError(f"Job failed: {status.get('error')}")
    time.sleep(5)

# 3. Download results
result = requests.get(f"{BASE}/result/{job['jobId']}/", headers=HEADERS)
with open("output.csv", "wb") as f:
    f.write(result.content)
print("Saved to output.csv")
```

---

### Quick Start — curl

```bash
API_KEY="ak_your_key_here"
BASE="https://predictor.openkinetics.org/api/v1"

# 1. Submit
JOB=$(curl -s -X POST "$BASE/submit/" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@input.csv" \
  -F "predictionType=kcat" \
  -F "kcatMethod=DLKcat" \
  -F "handleLongSequences=truncate")

JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")
echo "Submitted: $JOB_ID"

# 2. Poll
while true; do
  STATE=$(curl -s "$BASE/status/$JOB_ID/" \
    -H "Authorization: Bearer $API_KEY" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  $STATE"
  [ "$STATE" = "Completed" ] && break
  [ "$STATE" = "Failed" ]    && { echo "Job failed"; exit 1; }
  sleep 5
done

# 3. Download
curl -s "$BASE/result/$JOB_ID/" \
  -H "Authorization: Bearer $API_KEY" \
  -o output.csv
```

---

### JSON Body Submission (no CSV file needed)

For small datasets (≤ 10,000 rows) you can send data directly as JSON:

```python
requests.post(
    f"{BASE}/submit/",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "predictionType":      "kcat",
        "kcatMethod":          "DLKcat",
        "handleLongSequences": "truncate",
        "useExperimental":     True,
        "data": [
            {"Protein Sequence": "MKTLLIFAG...", "Substrate": "CC(=O)O"},
            {"Protein Sequence": "MGSSHHHHH...", "Substrate": "C1CCCCC1"},
        ],
    },
)
```

---

### Validating Input Before Submission

Use `/validate/` to check substrate SMILES/InChI strings, protein sequences,
and per-model length limits **without consuming any quota or running predictions**.
This is equivalent to the validation step available in the web interface.

```python
import requests

API_KEY = "ak_your_key_here"
BASE    = "https://predictor.openkinetics.org/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Basic validation (fast)
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "false"},
    )

result = resp.json()
print(f"Rows: {result['rowCount']}")
print(f"Invalid substrates: {len(result['invalidSubstrates'])}")
print(f"Invalid proteins:   {len(result['invalidProteins'])}")
print(f"Length violations:  {result['lengthViolations']}")
```

Set `runSimilarity=true` to also run MMseqs2 sequence similarity analysis
against each method's training database. The request **blocks synchronously**
until the analysis is complete (can take several minutes for large inputs):

```python
with open("input.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "true"},
        timeout=600,
    )

similarity = resp.json()["similarity"]
for method, data in similarity.items():
    print(f"{method}: avg max identity = {data['average_max_similarity']:.1f}%")
```

The `similarity` field in the response is a dict keyed by method name. Each entry
contains `histogram_max`, `histogram_mean` (10-bin arrays, 0–100% identity),
`average_max_similarity`, `average_mean_similarity`, `count_max`, and `count_mean`.

---

### CSV Format

Supported input schemas are:

- `Protein Sequence, Substrate` for one molecular input.
- `Protein Sequence, Substrates` for a Multi-Substrate input: an ordered,
  semicolon-separated list.
- `Protein Sequence, Substrates, Products` for a full reaction. `Products` is
  required by TurNup and is preserved but ignored by substrate-pair methods.

For a `Substrates` list, single-substrate methods predict every protein/substrate
pair. The reaction kcat is the maximum successful value; Km and direct kcat/Km
outputs are JSON arrays in substrate order, with `null` for failed entries.
The matching `Extra Info <target>` cell contains a JSON array with each
substrate's one-based position, input value, prediction, source, error, and the
selected kcat maximum.

CatPred kcat is the exception: it consumes the complete ordered substrate set
as one native multi-substrate input and returns one scalar. CatPred Km still
uses the per-substrate array behavior, including when kcat and Km are requested
together. TurNup consumes both sides of a full reaction; every other method
preserves but ignores `Products`.

| Method | Predicts | Required columns | Max sequence length |
| ------ | -------- | ---------------- | ------------------- |
| DLKcat | kcat | `Protein Sequence`, `Substrate` or `Substrates` | No limit |
| TurNup | kcat | `Protein Sequence`, `Substrates`, `Products` | 1,024 residues |
| EITLEM | kcat or Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,024 residues |
| UniKP | kcat or Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,000 residues |
| CataPro | kcat, Km, or kcat/Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,000 residues |
| KinForm-H | kcat or Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,500 residues |
| KinForm-L | kcat only | `Protein Sequence`, `Substrate` or `Substrates` | 1,500 residues |
| CatPred | kcat or Km | kcat: `Substrates`; Km: `Substrate` or `Substrates` | 2,048 residues |
| OmniESI | kcat or Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,000 residues |
| RealKcat | kcat or Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,022 residues |
| IECata | kcat/Km | `Protein Sequence`, `Substrate` or `Substrates` | 1,000 residues |
| MMISA-KM | Km | `Protein Sequence`, `Substrate` or `Substrates` | 500 residues |

Substrates and products must be SMILES or InChI strings. Separate ordered
values in `Substrates` and `Products` with semicolons, for example
`CC(=O)O;C1CCCCC1`. Supplied products are validated even when the selected
method preserves but does not use them.

---

### Rate Limits

* **20,000 input reaction rows/day** per API key (default; custom limits available).
  Internal per-substrate predictions do not consume additional quota.
* Counter resets at midnight UTC.
* Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
* HTTP `429` is returned when the quota is exceeded.

---

### Error Format

All errors return JSON with a single `error` key:

```json
{ "error": "A human-readable description of what went wrong." }
```

| Status | Meaning |
| ------ | ------- |
| 400 | Invalid parameters, missing CSV columns, or bad data |
| 401 | Missing or invalid API key |
| 403 | Account suspended |
| 404 | Job not found |
| 409 | Results not ready yet |
| 429 | Quota exceeded |
| 500 | Internal server error |

---

## Attribution

Please cite the original publications when using predictions from a specific engine. Cite all underlying sources plus this platform.

## Contact

For questions or collaboration: open an issue or reach out to the authors of the respective model.

## Funding

This work was supported by EU Horizon Europe #101080997, Swiss SERI #23.00232, UKRI #10083717 & #10080153, FNR PRIDE21/16763386/CANBIO2, Novo Nordisk Foundation #NNF10CC1016517, Knut & Alice Wallenberg Foundation, EU Horizon 2020 #686070 & #814650, National Key R&D China 2025YFA0922700
