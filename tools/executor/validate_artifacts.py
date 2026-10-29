"""Validate generated representations in the model's existing Python environment."""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def metadata(path, request):
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype.kind == "U":
            if request["payload"].get("molecular_method") != "CatRange" or value.shape != (1,):
                raise ValueError("Unexpected textual embedding")
            return {"shape": [1], "dtype": str(value.dtype), "input_rejection": True}
        if not value.size or value.dtype.kind != "f":
            raise ValueError("Embedding is empty or has a non-floating dtype")
        flat = value.reshape(-1)
        if not all(np.isfinite(flat[i : i + 1024**2]).all() for i in range(0, flat.size, 1024**2)):
            raise ValueError("Non-finite embedding")
        return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
    if path.suffix == ".pt":
        import torch

        try:
            value = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # Original supported PyTorch environments predate weights_only.
            value = torch.load(str(path), map_location="cpu")
        if (
            not torch.is_tensor(value)
            or not value.numel()
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Invalid generated tensor")
        return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
    if path.suffix == ".tsv" and request["payload"].get("step") in {
        "kinform_pseq2sites",
        "kinform_binding_merge",
    }:
        wanted = request["payload"]["seq_id_to_seq"]
        seen = set()
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sid = row["PDB"]
                values = np.asarray([float(value) for value in row["Pred_BS_Scores"].split(",")])
                if (
                    sid not in wanted
                    or sid in seen
                    or len(values) != min(1024, len(wanted[sid].replace(",", "")))
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid binding-site representation")
                seen.add(sid)
        if seen != set(wanted):
            raise ValueError("Missing binding-site rows")
        return {"rows": len(seen), "dtype": "binding_site_scores", "finite": True}
    raise ValueError("Unknown embedding artifact format")


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    media = Path(sys.argv[2])
    result = {
        output["id"]: metadata(media / output["path"], request)
        for output in request["payload"]["outputs"]
    }
    Path(sys.argv[3]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
