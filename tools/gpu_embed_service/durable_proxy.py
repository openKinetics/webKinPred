import json
import os
import urllib.error
import urllib.request


def enabled():
    return os.getenv("GPU_EMBED_DURABLE_PROXY", "0") == "1"


def call(operation, data=None):
    url = os.environ["GPU_EMBED_CONTROL_URL"].rstrip("/") + "/internal/v2/executors/gpu/" + operation
    request = urllib.request.Request(url, data=json.dumps(data or {}).encode(), headers={"Authorization": "Bearer " + os.environ["GPU_EXECUTOR_CONTROL_TOKEN"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Durable GPU scheduling is temporarily unavailable; retry with the same request.") from error
