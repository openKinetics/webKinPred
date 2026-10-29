# Active measured resource profiles — 29 October 2026

These 86 profiles were exported from the live validated registry after rollout.
All have at least three fresh and five warm runs. Their full image IDs, runtime
identities, dataset checksums and raw measurements remain in the private registry
and profiling evidence. GPU numerical parity was waived by the user; CPU
prediction compatibility and embedding shape/dtype/finiteness remain required.

Reservations include measured margins: RAM adds the larger of 25% or 512 MiB;
VRAM adds the larger of 25% or 256 MiB; temporary disk adds the larger of 25% or
1 GiB. Disk values below are base envelopes; admission additionally reserves twice
the input-artifact bytes for transfer and private working copies. Host headroom,
external activity and unspent reservations also apply. CPU and GPU work share one
host budget. A profile's presence does not authorize an unmeasured GPU combination.

Lengths and unit counts use each adapter's declared/preprocessed shape metrics.
The microbatch column differs from a stage's validated sequence/unit limit.
Unknown shapes wait for profiling. CPU prediction profiles on the GPU host have
zero VRAM and receive no GPU device access.

## Production

| Profile | Stage / step | Targets | CPUs | RAM GiB | VRAM GiB | Disk GiB | Microbatch | Validated shape limits |
|---|---|---|---:|---:|---:|---:|---:|---|
| production34-assembly | assemble |  | 0.5 | 0.67 | 0.00 | 1.19 | — | rows≤10000 |
| production29-CatPred-Km | predict | Km | 1.0 | 2.05 | 0.00 | 1.02 | 128 | input_bytes≤558374, max_length≤2048, max_molecule_length≤257, rows≤128, units≤256 |
| production29-CatPred-kcat | predict | kcat | 1.0 | 1.99 | 0.00 | 1.02 | 128 | input_bytes≤558374, max_length≤2048, max_molecule_length≤257, rows≤128, units≤256 |
| production29-CatRange-Km | predict | Km | 1.0 | 4.99 | 0.00 | 1.00 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| production29-CatRange-kcat | predict | kcat | 1.0 | 5.33 | 0.00 | 1.00 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| production16-CataPro-predict-Km | predict | Km | 1.0 | 1.24 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| production16-CataPro-predict-kcat | predict | kcat | 1.0 | 1.10 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| production16-CataPro-predict-kcat_Km | predict | kcat/Km | 1.0 | 1.14 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| production16-DLKcat-predict-kcat | predict | kcat | 1.0 | 0.98 | 0.00 | 1.01 | 128 | input_bytes≤2594086, max_length≤10000, max_molecule_length≤257, rows≤128, units≤256 |
| production29-OmniESI-Km | predict | Km | 1.0 | 2.23 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| production29-OmniESI-O2DENet-Km | predict | Km | 1.0 | 1.85 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| production16-CatPred-similarity | similarity | kcat | 1.0 | 1.60 | 0.00 | 1.28 | — | rows≤10000 |
| production16-CatRange-similarity | similarity | kcat | 1.0 | 0.79 | 0.00 | 1.28 | — | rows≤10000 |
| production16-CataPro-similarity | similarity | kcat | 1.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |
| production16-DLKcat-similarity | similarity | kcat | 1.0 | 1.61 | 0.00 | 1.28 | — | rows≤10000 |
| production16-EITLEM-similarity | similarity | kcat | 1.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |
| production16-IECata-similarity | similarity | kcat | 1.0 | 1.59 | 0.00 | 1.28 | — | rows≤10000 |
| production16-KinForm-H-similarity | similarity | kcat | 1.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |
| production16-KinForm-L-similarity | similarity | kcat | 1.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |
| production16-MMISA-KM-similarity | similarity | kcat | 1.0 | 0.72 | 0.00 | 1.28 | — | rows≤10000 |
| production16-OmniESI-similarity | similarity | kcat | 1.0 | 1.60 | 0.00 | 1.28 | — | rows≤10000 |
| production16-OmniESI-O2DENet-similarity | similarity | kcat | 1.0 | 1.60 | 0.00 | 1.28 | — | rows≤10000 |
| production16-TurNup-similarity | similarity | kcat | 1.0 | 1.58 | 0.00 | 1.28 | — | rows≤10000 |
| production16-UniKP-similarity | similarity | kcat | 1.0 | 1.61 | 0.00 | 1.28 | — | rows≤10000 |

## GPU host

| Profile | Stage / step | Targets | CPUs | RAM GiB | VRAM GiB | Disk GiB | Microbatch | Validated shape limits |
|---|---|---|---:|---:|---:|---:|---:|---|
| gpu16-CatPred-catpred_embed-Km | embedding/catpred_embed | Km | 2.0 | 6.77 | 4.79 | 1.00 | 1 | max_length≤2048, sequences≤16 |
| gpu16-CatPred-catpred_embed-kcat | embedding/catpred_embed | kcat | 2.0 | 6.77 | 4.79 | 1.00 | 1 | max_length≤2048, sequences≤16 |
| gpu16-CatRange-catrange_esmc | embedding/catrange_esmc | kcat, Km, kcat/Km | 2.0 | 2.26 | 5.94 | 1.00 | 1 | max_length≤1024, sequences≤16 |
| gpu16-CataPro-prot_t5_mean | embedding/prot_t5_mean | kcat, Km, kcat/Km | 2.0 | 12.56 | 4.91 | 1.00 | 1 | max_length≤1000, sequences≤16 |
| gpu16-EITLEM-eitlem_esm1v | embedding/eitlem_esm1v | kcat, Km, kcat/Km | 2.0 | 12.65 | 3.80 | 1.08 | 1 | max_length≤1024, sequences≤16 |
| gpu16-IECata-iecata_prot_t5_residues | embedding/iecata_prot_t5_residues | kcat, Km, kcat/Km | 2.0 | 12.32 | 4.91 | 1.06 | 1 | max_length≤1000, sequences≤16 |
| gpu16-KinForm-H-kinform_esm2_residues | embedding/kinform_esm2_residues | kcat, Km, kcat/Km | 2.0 | 6.60 | 4.28 | 1.23 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-H-kinform_esmc_residues | embedding/kinform_esmc_residues | kcat, Km, kcat/Km | 2.0 | 2.25 | 6.24 | 1.21 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-H-kinform_pseq2sites | embedding/kinform_pseq2sites | kcat, Km, kcat/Km | 2.0 | 4.19 | 1.15 | 1.09 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-H-kinform_t5_residues | embedding/kinform_t5_residues | kcat, Km, kcat/Km | 2.0 | 12.31 | 5.63 | 1.18 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-L-kinform_esm2_residues | embedding/kinform_esm2_residues | kcat, Km, kcat/Km | 2.0 | 6.60 | 4.28 | 1.23 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-L-kinform_esmc_residues | embedding/kinform_esmc_residues | kcat, Km, kcat/Km | 2.0 | 2.25 | 6.24 | 1.21 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-L-kinform_pseq2sites | embedding/kinform_pseq2sites | kcat, Km, kcat/Km | 2.0 | 4.19 | 1.15 | 1.09 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-L-kinform_t5_residues | embedding/kinform_t5_residues | kcat, Km, kcat/Km | 2.0 | 12.31 | 5.63 | 1.18 | 1 | max_length≤1500, sequences≤16 |
| gpu16-OmniESI-omniesi_esm2 | embedding/omniesi_esm2 | kcat, Km, kcat/Km | 2.0 | 6.40 | 4.67 | 1.08 | 1 | max_length≤1000, sequences≤16 |
| gpu16-OmniESI-O2DENet-omniesi_esm2 | embedding/omniesi_esm2 | kcat, Km, kcat/Km | 2.0 | 6.40 | 4.67 | 1.08 | 1 | max_length≤1000, sequences≤16 |
| gpu16-TurNup-turnup_esm1b | embedding/turnup_esm1b | kcat, Km, kcat/Km | 2.0 | 15.82 | 3.81 | 1.00 | 1 | max_length≤1022, sequences≤16 |
| gpu16-UniKP-prot_t5_mean | embedding/prot_t5_mean | kcat, Km, kcat/Km | 2.0 | 12.56 | 4.91 | 1.00 | 1 | max_length≤1000, sequences≤16 |
| gpu16-CatRange-molecular | embedding/molecular | kcat, Km, kcat/Km | 2.0 | 1.53 | 0.81 | 1.00 | 1 | max_length≤512, sequences≤16 |
| gpu16-CataPro-molecular | embedding/molecular | kcat, Km, kcat/Km | 2.0 | 1.45 | 0.94 | 1.00 | 1 | max_length≤512, sequences≤16 |
| gpu17-KinForm-H-molecular | embedding/molecular | kcat, Km, kcat/Km | 2.0 | 1.25 | 0.50 | 1.00 | 1 | max_length≤512, sequences≤16 |
| gpu17-KinForm-L-molecular | embedding/molecular | kcat, Km, kcat/Km | 2.0 | 1.25 | 0.50 | 1.00 | 1 | max_length≤512, sequences≤16 |
| gpu16-UniKP-molecular | embedding/molecular | kcat, Km, kcat/Km | 2.0 | 3.25 | 0.89 | 1.00 | 1 | max_length≤512, sequences≤16 |
| gpu29-CatPred-Km | predict | Km | 2.0 | 1.96 | 0.00 | 1.02 | 128 | input_bytes≤558374, max_length≤2048, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-CatPred-kcat | predict | kcat | 2.0 | 1.96 | 0.00 | 1.02 | 128 | input_bytes≤558374, max_length≤2048, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-CatRange-Km | predict | Km | 2.0 | 4.46 | 0.00 | 1.00 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-CatRange-kcat | predict | kcat | 2.0 | 4.90 | 0.00 | 1.00 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-CataPro-predict-Km | predict | Km | 2.0 | 1.08 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-CataPro-predict-kcat | predict | kcat | 2.0 | 1.06 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-CataPro-predict-kcat_Km | predict | kcat/Km | 2.0 | 1.11 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-DLKcat-predict-kcat | predict | kcat | 2.0 | 0.94 | 0.00 | 1.01 | 128 | input_bytes≤2594086, max_length≤10000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-EITLEM-predict-Km | predict | Km | 1.0 | 4.81 | 0.00 | 2.22 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-EITLEM-predict-kcat | predict | kcat | 1.0 | 4.81 | 0.00 | 2.22 | 128 | input_bytes≤296230, max_length≤1024, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-IECata-predict-kcat_Km | predict | kcat/Km | 2.0 | 4.26 | 0.00 | 1.98 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-KinForm-H-predict-Km | predict | Km | 2.0 | 1.39 | 0.00 | 1.02 | 128 | input_bytes≤418086, max_length≤1500, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-KinForm-H-predict-kcat | predict | kcat | 2.0 | 2.11 | 0.00 | 1.02 | 128 | input_bytes≤418086, max_length≤1500, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-KinForm-L-predict-kcat | predict | kcat | 2.0 | 2.17 | 0.00 | 1.02 | 128 | input_bytes≤418086, max_length≤1500, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-KinForm-H-kinform_binding_merge | predict/kinform_binding_merge | kcat, Km | 1.0 | 0.56 | 0.00 | 1.01 | 1 | max_length≤1500, sequences≤256 |
| gpu16-KinForm-L-kinform_binding_merge | predict/kinform_binding_merge | kcat, Km | 1.0 | 0.56 | 0.00 | 1.01 | 1 | max_length≤1500, sequences≤256 |
| gpu16-KinForm-H-kinform_pool | predict/kinform_pool | kcat, Km | 1.0 | 1.09 | 0.00 | 1.62 | 1 | max_length≤1500, sequences≤16 |
| gpu16-KinForm-L-kinform_pool | predict/kinform_pool | kcat, Km | 1.0 | 1.09 | 0.00 | 1.62 | 1 | max_length≤1500, sequences≤16 |
| gpu16-MMISA-KM-predict-Km | predict | Km | 2.0 | 1.14 | 0.00 | 1.00 | 128 | input_bytes≤162086, max_length≤500, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-OmniESI-Km | predict | Km | 2.0 | 2.15 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-OmniESI-kcat | predict | kcat | 2.0 | 2.21 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-OmniESI-O2DENet-Km | predict | Km | 2.0 | 2.04 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu29-OmniESI-O2DENet-kcat | predict | kcat | 2.0 | 2.07 | 0.00 | 2.22 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-TurNup-predict-kcat | predict | kcat | 2.0 | 0.99 | 0.00 | 1.00 | 128 | input_bytes≤295718, max_length≤1022, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-UniKP-predict-Km | predict | Km | 2.0 | 1.28 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-UniKP-predict-kcat | predict | kcat | 2.0 | 1.38 | 0.00 | 1.00 | 128 | input_bytes≤290086, max_length≤1000, max_molecule_length≤257, rows≤128, units≤256 |
| gpu16-CatPred-similarity | similarity | kcat | 2.0 | 1.62 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-CatRange-similarity | similarity | kcat | 2.0 | 0.69 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-CataPro-similarity | similarity | kcat | 2.0 | 1.65 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-DLKcat-similarity | similarity | kcat | 2.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-EITLEM-similarity | similarity | kcat | 2.0 | 1.65 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-IECata-similarity | similarity | kcat | 2.0 | 1.61 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-KinForm-H-similarity | similarity | kcat | 2.0 | 1.65 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-KinForm-L-similarity | similarity | kcat | 2.0 | 1.65 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-MMISA-KM-similarity | similarity | kcat | 2.0 | 0.69 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-OmniESI-similarity | similarity | kcat | 2.0 | 1.62 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-OmniESI-O2DENet-similarity | similarity | kcat | 2.0 | 1.62 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-TurNup-similarity | similarity | kcat | 2.0 | 1.58 | 0.00 | 1.28 | — | rows≤10000 |
| gpu16-UniKP-similarity | similarity | kcat | 2.0 | 1.63 | 0.00 | 1.28 | — | rows≤10000 |

See [the deployment record](execution-deployment-2026-10-29.md) for admission ceilings, successful live overlap and measured serial/concurrent throughput.
