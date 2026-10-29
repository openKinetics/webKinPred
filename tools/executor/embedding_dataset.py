"""Create reproducible isolated shape probes from a known-working stage request."""

import copy
import hashlib

from api.execution.artifacts import atomic_json


def create(template, destination, *, count, length, batch_size=1, mixed=False):
    if not 1 <= count <= 16 or not 1 <= length <= 8192 or batch_size < 1:
        raise ValueError("Invalid bounded embedding probe dimensions")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "media").mkdir()
    (destination / "tmp").mkdir()
    request = copy.deepcopy(template)
    request["settings"]["batch_size"] = batch_size
    payload = request["payload"]
    outputs = payload["outputs"]
    payload["shape"] = {"sequences": count, "max_length": length}
    if payload["step"] == "molecular":
        method = payload["molecular_method"]
        positions = [0, 1, 99, 0, 27, 0]
        payload["rows"] = [
            {
                "input": "C" * (min(length, 3 + index) if mixed else length),
                "legacy_position": positions[index % len(positions)],
                "path": "sequence_info/molecular/" + method + "/probe" + str(index) + ".npy",
            }
            for index in range(count)
        ]
        paths = [row["path"] for row in payload["rows"]]
        payload.pop("seq_id_to_seq", None)
    else:
        old_id = next(iter(payload["seq_id_to_seq"]))
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        mapping = {}
        for index in range(count):
            size = max(1, length - index * max(1, length // count)) if mixed else length
            rotation = alphabet[index % len(alphabet) :] + alphabet[: index % len(alphabet)]
            sequence = ("M" + rotation * (size // len(rotation) + 1))[:size]
            sid = "probe_" + hashlib.sha256(sequence.encode()).hexdigest()[:20]
            mapping[sid] = sequence
        if len(mapping) != count:
            raise ValueError("Probe dimensions do not produce distinct sequences")
        payload["seq_id_to_seq"] = mapping
        paths = (
            [outputs[0]["path"]]
            if payload["step"] == "kinform_pseq2sites"
            else [output["path"].replace(old_id, sid) for sid in mapping for output in outputs]
        )
    payload["outputs"] = [
        {"id": str(index), "identity": {"probe": True, "path": path}, "path": path}
        for index, path in enumerate(paths)
    ]
    atomic_json(destination / "request.json", request)
    return request
