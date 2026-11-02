"""Generate private deployment templates without retaining SSH passwords."""

import argparse
import json
import os
import secrets
from pathlib import Path

from api.execution.resources import GIB


def generate(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    nodes = {"production": "100.125.248.109", "gpu": "100.78.161.61"}
    tokens = {host: secrets.token_hex(32) for host in nodes}
    artifact_tokens = {host: secrets.token_hex(32) for host in nodes}
    hosts = {}
    for host, address in nodes.items():
        budgets = {
            "cpu": 3 if host == "production" else 16,
            "ram": (10 if host == "production" else 96) * GIB,
            "disk": (8 if host == "production" else 128) * GIB,
            "ram_headroom": (4 if host == "production" else 16) * GIB,
            "disk_headroom": (15 if host == "production" else 50) * GIB,
            "gpu_slots": 1,
            "gpu_cpu_reserve": 0 if host == "production" else 4,
        }
        config = {
            "host_id": host,
            "control_url": "http://100.125.248.109:8000",
            "token": tokens[host],
            "artifact_token": artifact_tokens[host],
            "root": "/var/lib/webkinpred",
            "bind": address,
            "artifact_port": 8082,
            "budgets": budgets,
            "capabilities": {"profiles": [], "validated_gpu_combinations": []},
            "images": [],
            "runtime_files": {},
            "model_mounts": [],
            "image_mounts": {},
            "gpu_uuid": "GPU-0eda6548-3c40-c109-c01d-5b9402d01ae8" if host == "gpu" else "",
            "gpu_runtime": "nvidia" if host == "gpu" else "",
            "peers": {
                peer: {"url": "http://" + ip + ":8082", "token": artifact_tokens[peer]}
                for peer, ip in nodes.items()
                if peer != host
            },
        }
        path = directory / (host + ".json")
        path.write_text(json.dumps(config, indent=2))
        os.chmod(path, 0o600)
        hosts[host] = {"token": tokens[host], "budgets": budgets}
    password = secrets.token_hex(24)
    metrics_token = secrets.token_hex(32)
    (directory / "metrics-token").write_text(metrics_token)
    os.chmod(directory / "metrics-token", 0o600)
    env = {
        "EXECUTION_V2_ENABLED": "0",
        "EXECUTION_START_PAUSED": "1",
        "EXECUTION_DB_PASSWORD": password,
        "EXECUTION_METRICS_TOKEN": metrics_token,
        "EXECUTION_HOSTS_JSON": json.dumps(hosts, separators=(",", ":")),
        "EXECUTION_MODEL_VERSIONS_JSON": "{}",
        "EXECUTION_STAGE_VERSIONS_JSON": "{}",
        "EXECUTION_ALLOWED_HOSTS": "100.125.248.109",
        "DATABASE_URL": f"postgresql://webkinpred:{password}@execution-db:5432/webkinpred",
        "SEQMAP_DATABASE_URL": f"postgresql://webkinpred:{password}@execution-db:5432/webkinpred_seqmap",
        "PREDICTION_STORE_DATABASE_URL": f"postgresql://webkinpred:{password}@execution-db:5432/webkinpred_predictions",
        "EXECUTION_CONFIG_DIR": str(directory.resolve()),
        "EXECUTION_MODEL_ROOT": "/var/lib/webkinpred/models",
        "EXECUTION_GPU_MODEL_ROOT": "/var/lib/webkinpred/gpu-models",
        "EXECUTION_GPU_UUID": "GPU-0eda6548-3c40-c109-c01d-5b9402d01ae8",
        "EXECUTION_GPU_RUNTIME": "nvidia",
        "EXECUTION_WEB_IMAGE": "UNCONFIGURED",
        "EXECUTION_FRONTEND_IMAGE": "UNCONFIGURED",
        "EXECUTION_AGENT_IMAGE": "UNCONFIGURED",
        "EXECUTION_CLEANUP_IMAGE": "UNCONFIGURED",
        "EXECUTION_PROMETHEUS_IMAGE": "UNCONFIGURED",
        "EXECUTION_GRAFANA_IMAGE": "UNCONFIGURED",
        "EXECUTION_GRAFANA_PASSWORD": secrets.token_hex(24),
    }
    path = directory / "execution.env"
    path.write_text("\n".join(key + "='" + value + "'" for key, value in env.items()) + "\n")
    os.chmod(path, 0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
