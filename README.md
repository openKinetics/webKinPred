# WebKinPred

WebKinPred is a production web interface for predicting enzyme kinetic parameters (kcat and KM) from protein sequence and substrate SMILES. It consolidates several state‑of‑the‑art machine learning / deep learning models behind a unified, asynchronous job API so you can submit sequences and retrieve structured predictions.

**Live service:** [https://kineticxpredictor.humanmetabolism.org/](https://kineticxpredictor.humanmetabolism.org/)

## Prediction Engines

| Engine | Input needed | Output | Citation |
|--------|--------------|--------|----------|
| KinForm-H | Protein sequence + substrate SMILES | kcat or KM | [Alwer & Fleming, Arxiv](https://arxiv.org/abs/2507.14639) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| KinForm-L | Protein sequence + substrate SMILES | kcat or KM | [Alwer & Fleming, Arxiv](https://arxiv.org/abs/2507.14639) ([GitHub](https://github.com/Digital-Metabolic-Twin-Centre/KinForm)) |
| UniKP | Protein sequence + substrate SMILES | kcat or KM | [Yu et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-44113-1) ([GitHub](https://github.com/Luo-SynBioLab/UniKP)) |
| DLKcat | Protein sequence + substrate SMILES | kcat | [Li et al., Nat Catal 2022](https://www.nature.com/articles/s41929-022-00798-z) ([GitHub](https://github.com/SysBioChalmers/DLKcat)) |
| TurNup | Protein sequence + substrates list + products list | kcat | [Kroll et al., Nat Commun 2023](https://www.nature.com/articles/s41467-023-39840-4) ([GitHub](https://github.com/AlexanderKroll/Kcat_prediction)) |
| EITLEM | Protein sequence + substrate SMILES | kcat or KM | [Shen et al., Biotechnol Adv 2024](https://www.sciencedirect.com/science/article/pii/S2667109324002665) ([GitHub](https://github.com/XvesS/EITLEM-Kinetics)) |

Each model is loaded with its published weights/code (see `api/prediction_engines/`) and invoked through a standard internal interface so new engines can be added with minimal wiring.

## Features

* Batch submission of sequences and substrates.
* Long‑running inference handled asynchronously (Celery + Redis) with progress tracking.
* Sequence similarity distribution of input data vs mehtods' training data (Using mmseq2).
* Caching sequence embeddings.

## Tech Stack

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

## High-Level Flow

1. User submits a job (sequence + substrate(s) [+ products/mutant context if required]) via the frontend.
2. Backend validates input (`api/services/validation_service.py`).
3. A Celery task is enqueued; Redis broker stores the task message.
4. Worker loads the selected model wrapper (e.g. `prediction_engines/kinform.py`) and executes inference.
5. Results & intermediate status are persisted; cached for repeated identical queries.
6. Frontend polls job status endpoint to update progress and results.

## Adding a New Model

... todo ...
Create a new module in `api/prediction_engines/` following existing wrappers.

## Attribution

Please cite the original publications when using predictions from a specific engine. When publishing aggregated results produced through WebKinPred, cite all underlying sources plus this platform.

## Contact

For questions or collaboration: open an issue or reach out to the authors of the respective model.

## Funding

... todo ...
