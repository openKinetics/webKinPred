"""Isolated legacy CPU versus position-preserving CUDA transformer probe."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from tools.executor.molecular import legacy_transformer_vectors, module_at


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compute-dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    if args.device == "cuda":
        from tools.executor.policy import require_cuda

        require_cuda(torch)
        # Contain probe allocations; never intentionally exhaust a shared GPU.
        torch.cuda.set_per_process_memory_fraction(0.075)
    torch.set_num_threads(2)
    os.environ["UNIKP_MEDIA_PATH"] = "/work/media"
    os.environ["UNIKP_TOOLS_PATH"] = "/app/tools"
    module = module_at(Path("/app/models/UniKP-main/run_unikp_batch.py"), "probe_unikp")
    vocab, model = module.load_smiles_model()
    model = model.to(args.device).eval()
    if args.compute_dtype == "float64":
        model = model.double()
    rows = json.loads(Path("/work/rows.json").read_text())
    timings = []
    for _ in range(args.repeats):
        start = time.monotonic()
        if args.device == "cpu":
            vectors = np.concatenate(
                [
                    module.smiles_to_vec([row["input"] for row in rows[i : i + 128]], vocab, model)
                    for i in range(0, len(rows), 128)
                ]
            )
        else:
            vectors = np.concatenate(
                [
                    legacy_transformer_vectors(
                        model, vocab, module.split, rows[i : i + args.batch_size], torch
                    )
                    for i in range(0, len(rows), args.batch_size)
                ]
            )
            torch.cuda.synchronize()
        timings.append(time.monotonic() - start)
    np.save("/work/vectors.npy", vectors.astype(np.float32))
    Path("/work/timing.json").write_text(
        json.dumps(
            {
                "device": args.device,
                "compute_dtype": args.compute_dtype,
                "seconds": timings,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "allocated_peak": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                "reserved_peak": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
            }
        )
    )


if __name__ == "__main__":
    main()
