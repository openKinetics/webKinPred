"""One trusted local supervisor; model processes never hold the queue or lease ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
from api.execution.artifacts import ArtifactStore, atomic_json, safe_relative, sha256_file
from api.execution.errors import MissingEmbedding, classify
from api.execution.resources import admit
from api.execution.telemetry import container_observations, snapshot

from tools.executor.checkpoints import partial_manifest, remember, restore

LABEL = "org.webkinpred.executor"


def file_stamp(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def directory_bytes(root):
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue  # A worker may atomically replace a temporary output.
    return total


class Agent:
    def __init__(self, config):
        self.config = config
        self.config.setdefault("capabilities", {})["artifact_cleanup_version"] = 2
        self.host = config["host_id"]
        self.root = Path(config["root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.store = ArtifactStore(self.root / "blobs")
        self.store.root.mkdir(exist_ok=True)
        self.jobs = {}
        self.last_contact = time.monotonic()
        self.last_cleanup = 0
        self.last_error_log = 0
        self.transfers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="artifact-transfer")

    def verify_runtime_files(self, threads=1):
        """Cache verified digests against inode, size, mtime and ctime, never mtime alone."""
        path = self.root / "model-validation.json"
        try:
            cache = json.loads(path.read_text())
            previous = cache.get("files", {}) if cache.get("version") == 1 else {}
        except (OSError, ValueError, AttributeError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        files = self.config.get("runtime_files", {})
        stamps = {name: file_stamp(Path(name)) for name in files}
        required = {}
        for name, checksum in files.items():
            cached = previous.get(name, {})
            if not isinstance(cached, dict):
                cached = {}
            if cached.get("stamp") != stamps[name] or cached.get("sha256") != checksum:
                required.setdefault(tuple(stamps[name]), name)

        def verify(item):
            stamp, name = item
            checksum = sha256_file(Path(name))
            if tuple(file_stamp(Path(name))) != stamp:
                raise ValueError("Runtime/model changed during verification")
            return stamp, checksum

        with ThreadPoolExecutor(max_workers=max(1, min(int(threads), 8))) as pool:
            measured = dict(pool.map(verify, required.items()))
        for name, checksum in files.items():
            if tuple(stamps[name]) in measured and measured[tuple(stamps[name])] != checksum:
                raise ValueError("Runtime/model checksum mismatch; executor remains offline")
            if file_stamp(Path(name)) != stamps[name]:
                raise ValueError("Runtime/model changed during verification")
        atomic_json(
            path,
            {
                "version": 1,
                "files": {
                    name: {"stamp": stamps[name], "sha256": checksum}
                    for name, checksum in files.items()
                },
            },
        )
        return stamps

    def request(self, operation, data=None, attempt=None):
        base = self.config["control_url"].rstrip("/") + "/internal/v2/executors/" + self.host
        url = base + (f"/attempts/{attempt}/{operation}" if attempt else "/" + operation)
        request = urllib.request.Request(
            url,
            data=json.dumps(data or {}).encode(),
            headers={
                "Authorization": "Bearer " + self.config["token"],
                "Content-Type": "application/json",
            },
        )
        timeout = 5
        if operation == "claim":
            # Allow a bounded cold admission without extending any compute
            # lease. Leave at least five seconds before local fencing is due.
            oldest = max(
                (time.monotonic() - job["renewed"] for job in self.jobs.values()), default=0
            )
            timeout = max(1, min(8, 15 - oldest))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        self.last_contact = time.monotonic()
        return result

    def docker(self, *args):
        return subprocess.check_output(
            ["docker", *args], text=True, stderr=subprocess.PIPE, timeout=20
        ).strip()

    def error_details(self, error, request):
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            message = str(error.stderr)
        secrets = [self.config.get("token"), request.get("token")]
        secrets.extend(peer.get("token") for peer in self.config.get("peers", {}).values())
        for secret in secrets:
            if secret:
                message = message.replace(str(secret), "[redacted]")
        return {"error_type": type(error).__name__, "message": message[:2000]}

    def retain_image(self, image):
        """Keep approved runtimes in use even between independently run attempts."""
        self.docker("image", "inspect", image)
        name = "wk-image-keeper-" + image.split(":")[-1][:16]
        existing = self.docker("ps", "-aq", "--filter", "name=^/" + name + "$")
        if existing:
            info = json.loads(self.docker("inspect", name))[0]
            if (
                info["Image"] != image
                or info["Config"]["Labels"].get("org.webkinpred.image-keeper") != "true"
            ):
                raise ValueError("Runtime image keeper ownership conflict")
            if not info["State"]["Running"]:
                self.docker("start", name)
            return
        self.docker(
            "run",
            "-d",
            "--name",
            name,
            "--restart",
            "unless-stopped",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "16m",
            "--cpus",
            "0.01",
            "--pids-limit",
            "16",
            "--label",
            "org.webkinpred.image-keeper=true",
            "--label",
            LABEL + "=",
            "--label",
            "com.docker.compose.project=webkinpred-image-keepers",
            "--label",
            "com.docker.compose.service=image-keeper",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "exec sleep infinity",
        )

    def inventory(self):
        names = self.docker("ps", "-aq", "--filter", f"label={LABEL}={self.host}").splitlines()
        return json.loads(self.docker("inspect", *names)) if names else []

    def stop_recovered(self):
        """Fence old attempts locally before network calls or lengthy file checks."""
        stopped = {}
        for info in self.inventory():
            attempt_id = info["Config"]["Labels"]["org.webkinpred.attempt"]
            path = self.root / "attempts" / attempt_id / "request.json"
            if not path.exists():
                # Unknown labelled workloads cannot be adopted with guessed ownership.
                self.docker("stop", "-t", "5", info["Id"])
                continue
            if info["State"]["Running"]:
                # A restarted agent cannot prove the previous monotonic lease deadline.
                self.docker("stop", "-t", "5", info["Id"])
            remember(self.root, json.loads(path.read_text()))
            state = json.loads(self.docker("inspect", info["Id"]))[0]["State"]
            stopped[attempt_id] = state["FinishedAt"]
        return stopped

    def recover(self):
        stopped = self.stop_recovered()
        # Inventory is complete before the control plane may release old reservations.
        # All labelled containers are now confirmed stopped; abandoned stages retry.
        self.request("reconcile", {"running": [], "stopped": stopped})

    def reconcile_unreceived(self, reply):
        """Recover a claim committed after its response was lost or timed out.

        The controller's lease is evidence of reservation, not of local launch.
        Inspect and fence unknown owned containers before releasing anything.
        Known transfers and running attempts retain their reservations.
        """
        unknown = set(reply.get("owned_attempts", [])) - self.jobs.keys()
        if not unknown:
            return
        stopped = {}
        for info in self.inventory():
            ident = info["Config"]["Labels"]["org.webkinpred.attempt"]
            if ident in self.jobs:
                continue
            if info["State"]["Running"]:
                self.docker("stop", "-t", "5", info["Id"])
            # Re-read termination after stop; failures preserve reservations.
            state = json.loads(self.docker("inspect", info["Id"]))[0]["State"]
            if state["Running"]:
                raise RuntimeError("Unknown owned attempt has not stopped")
            path = self.root / "attempts" / ident / "request.json"
            if path.exists():
                remember(self.root, json.loads(path.read_text()))
            stopped[ident] = state["FinishedAt"]
        self.request("reconcile", {"running": list(self.jobs), "stopped": stopped})
        print(
            json.dumps(
                {
                    "event": "executor.claim_response_recovered",
                    "host": self.host,
                    "unreceived_claims": len(unknown),
                }
            ),
            flush=True,
        )

    def materialize(self, request, work):
        started = time.monotonic()
        transferred_bytes, local_cache_hits = 0, 0
        for artifact in request["artifacts"]:
            local_location_invalidated = False
            destination = work / "media" / safe_relative(artifact["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob = self.store.path(artifact["checksum"])
            if blob.exists() and (
                blob.stat().st_size != artifact["size"] or sha256_file(blob) != artifact["checksum"]
            ):
                blob.unlink()  # A corrupt blob is never an approved dependency.
            if not blob.exists():
                error = None
                missing = False
                for host in artifact["hosts"]:
                    if host == self.host:
                        self.artifact_location(request, artifact, host, False)
                        local_location_invalidated = True
                        missing = True
                        continue
                    peer = self.config.get("peers", {}).get(host)
                    if not peer:
                        continue
                    try:
                        req = urllib.request.Request(
                            peer["url"].rstrip("/") + "/blobs/" + artifact["checksum"],
                            headers={"Authorization": "Bearer " + peer["token"]},
                        )
                        with urllib.request.urlopen(req, timeout=30) as response:
                            blob = self.store.publish(
                                response,
                                artifact["checksum"],
                                artifact["size"],
                                minimum_free=self.config["budgets"]["disk_headroom"],
                            )
                        transferred_bytes += artifact["size"]
                        break
                    except (OSError, ValueError) as exc:
                        error = exc
                        if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                            self.artifact_location(request, artifact, host, False)
                            missing = True
                else:
                    if missing:
                        raise MissingEmbedding("Required artifact must be rescheduled") from error
                    raise ConnectionError("Required artifact transfer unavailable") from error
            else:
                local_cache_hits += 1
            # Existing verified locations need no network write after the
            # local size/checksum check. Prediction stages can consume hundreds
            # of embeddings already produced on this host.
            if self.host not in artifact["hosts"] or local_location_invalidated:
                self.artifact_location(request, artifact, self.host, True)
            # Private copies prevent legacy writers from changing immutable cache files.
            shutil.copyfile(blob, destination)
        restore(self.root, request, work)
        return {
            "transfer_seconds": time.monotonic() - started,
            "transferred_bytes": transferred_bytes,
            "local_cache_hits": local_cache_hits,
            "artifact_requests": len(request["artifacts"]),
        }

    def artifact_location(self, request, artifact, host, ready):
        self.request(
            "artifact-location",
            {
                "token": request["token"],
                "artifact_id": artifact["id"],
                "checksum": artifact["checksum"],
                "size": artifact["size"],
                "source_host": host,
                "ready": ready,
            },
            request["id"],
        )

    def stage(self, request):
        ident = request["id"]
        work = self.root / "attempts" / ident
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        for folder in ["media", "tmp"]:
            (work / folder).mkdir(exist_ok=True)
        atomic_json(work / "request.json", request)
        self.jobs[ident] = {
            "request": request,
            "container": None,
            "renewed": time.monotonic(),
            "transfer": self.transfers.submit(self.materialize, request, work),
        }

    def launch(self, request):
        ident = request["id"]
        if request["image"] not in self.config["images"]:
            raise ValueError("Execution image is not locally approved")
        if "sha256:" not in request["image"]:
            raise ValueError("Execution image must be pinned by digest")
        if request["kind"] not in {"prepare", "embedding", "predict", "similarity", "assemble"}:
            raise ValueError("Unknown stage kind")
        work = self.root / "attempts" / ident
        limits = request["resources"]
        # Artifact transfer may take minutes. Recheck immediately before model
        # creation, excluding this attempt's already-held reservation.
        telemetry = self.observe()
        active = [
            {
                "resources": job["request"]["resources"],
                "observed": telemetry["attempts"].get(key, {}),
            }
            for key, job in self.jobs.items()
            if key != ident
        ]
        admission = admit(
            limits, active, self.config["budgets"], telemetry, gpu=bool(limits.get("vram"))
        )
        if not admission.fits:
            from api.execution.errors import RecoverableExecutionError

            error = RecoverableExecutionError("Capacity changed during artifact transfer")
            error.category = admission.reason
            raise error
        assigned = [cpu for value in self.jobs.values() for cpu in value.get("cpu_set", [])]
        cpu_set = sorted(
            psutil.Process().cpu_affinity(), key=lambda cpu: (assigned.count(cpu), cpu)
        )[: max(1, math.ceil(limits["cpu"]))]
        args = [
            "create",
            "--name",
            "wk-attempt-" + ident,
            "--label",
            f"{LABEL}={self.host}",
            "--label",
            "org.webkinpred.attempt=" + ident,
            "--label",
            "com.docker.compose.project=webkinpred-execution-attempts",
            "--label",
            "com.docker.compose.service=execution-attempt",
            "--init",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
            "--network",
            "none",
            "--memory",
            str(int(limits["ram"])),
            "--memory-swap",
            str(int(limits["ram"])),
            "--cpus",
            str(limits["cpu"]),
            "--cpuset-cpus",
            ",".join(map(str, cpu_set)),
            "--pids-limit",
            str(int(limits.get("pids", 128))),
            "--shm-size",
            str(min(int(limits["ram"]) // 4, 1024**3)),
            "--mount",
            f"type=bind,src={work},dst=/work",
            "--mount",
            f"type=bind,src={work / 'media'},dst=/app/media",
            "--env",
            "PYTHONPATH=/app",
            "--env",
            "TMPDIR=/work/tmp",
            "--env",
            "WEBKINPRED_ATTEMPT_FILE=/work/request.json",
            "--env",
            "WEBKINPRED_EXECUTION_POLICY=gpu-only",
            "--env",
            "WEBKINPRED_MEDIA_ROOT=/app/media",
            "--env",
            "DJANGO_SECRET_KEY=isolated-execution-no-http",
            "--env",
            "OMP_NUM_THREADS=" + str(max(1, int(limits["cpu"]))),
            "--env",
            "MKL_NUM_THREADS=" + str(max(1, int(limits["cpu"]))),
            "--env",
            "OPENBLAS_NUM_THREADS=" + str(max(1, int(limits["cpu"]))),
        ]
        if limits.get("vram"):
            if not self.config.get("gpu_uuid"):
                raise ValueError("GPU profile assigned to CPU-only host")
            if self.config.get("gpu_runtime"):
                args += [
                    "--runtime",
                    self.config["gpu_runtime"],
                    "--env",
                    "NVIDIA_VISIBLE_DEVICES=" + self.config["gpu_uuid"],
                    "--env",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                ]
            else:
                args += ["--gpus", "device=" + self.config["gpu_uuid"]]
            args += ["--env", "KINFORM_REQUIRE_CUDA=1"]
        else:
            args += ["--env", "CUDA_VISIBLE_DEVICES=", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
        mounts = [
            *self.config.get("model_mounts", []),
            *self.config.get("image_mounts", {}).get(request["image"], []),
        ]
        if len({mount["target"] for mount in mounts}) != len(mounts):
            raise ValueError("Ambiguous model mount destination")
        for mount in mounts:
            args += ["--mount", f"type=bind,src={mount['source']},dst={mount['target']},readonly"]
        args += [request["image"], "python", "-m", "tools.executor.runner", "/work/request.json"]
        try:
            container = self.docker(*args)
        except subprocess.CalledProcessError as creation_error:
            try:
                info = json.loads(self.docker("inspect", "wk-attempt-" + ident))[0]
            except subprocess.CalledProcessError:
                # Preserve the create error (e.g. a missing image) instead of
                # replacing it with the unhelpful 'no such container' error.
                raise creation_error from None
            if info["Config"]["Labels"].get("org.webkinpred.attempt") != ident:
                raise ValueError("Container ownership conflict") from None
            container = info["Id"]
        self.jobs[ident] = {
            "request": request,
            "container": container,
            "renewed": time.monotonic(),
            "cpu_set": cpu_set,
            "execution_metrics": self.jobs.get(ident, {}).get("execution_metrics", {}),
        }
        self.docker("start", container)
        try:
            from api.execution.telemetry import set_memory_high

            set_memory_high(container, limits["ram"])
        except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
            pass

    def observe(self):
        data = snapshot(self.root, self.config.get("gpu_uuid", ""))
        data["attempts"] = container_observations(
            {key: job["container"] for key, job in self.jobs.items() if job["container"]}
        )
        data["cpu_external"] = max(
            0, data["cpu_used"] - sum(value.get("cpu", 0) for value in data["attempts"].values())
        )
        rows = self.docker(
            "ps",
            "--filter",
            "label=com.docker.compose.project=" + self.config.get("legacy_project", "webkinpred"),
            "--format",
            '[{{json (.Label "com.docker.compose.service")}},'
            '{{json (.Label "org.webkinpred.profiling")}},'
            '{{json (.Label "org.webkinpred.executor")}}]',
        ).splitlines()
        # docker-committed runtime images retain their old Compose labels.
        # Identify our isolated probes/attempts before fencing legacy consumers.
        data["legacy_consumers"] = [
            service
            for service, profiling, executor in map(json.loads, rows)
            if service in {"celery", "celery-cache", "celery-beat"}
            and profiling != "true"
            and not executor
        ]
        return data

    def tick_jobs(self):
        for ident, job in list(self.jobs.items()):
            try:
                self.tick_job(ident, job)
            except urllib.error.HTTPError as exc:
                if exc.code not in {403, 409}:
                    raise
                # A stale result is never committed. Reconcile only after proving
                # this container stopped, retaining every other live reservation.
                future = next(
                    (
                        job[key]
                        for key in ("transfer", "publication")
                        if key in job and not job[key].done()
                    ),
                    None,
                )
                if future and not future.done():
                    job["abandoned"] = True
                    continue
                if job["container"]:
                    info = json.loads(self.docker("inspect", job["container"]))[0]
                    if info["State"]["Running"]:
                        self.docker("stop", "-t", "5", job["container"])
                del self.jobs[ident]
                self.request("reconcile", {"running": list(self.jobs)})
            except (OSError, ValueError) as exc:
                # Failed publication/transfer is recoverable, including a full spool.
                # Do not leave a stopped attempt in a permanently running state.
                if job["container"]:
                    info = json.loads(self.docker("inspect", job["container"]))[0]
                    if info["State"]["Running"]:
                        raise
                    remember(self.root, job["request"])
                    self.request(
                        "error",
                        {
                            "token": job["request"]["token"],
                            "stopped": True,
                            "category": classify(exc),
                        },
                        ident,
                    )
                    del self.jobs[ident]
                    self.docker("rm", job["container"])
                else:
                    raise

    def tick_job(self, ident, job):
        request = job["request"]
        if job.get("abandoned"):
            future = next(
                (
                    job[key]
                    for key in ("transfer", "publication")
                    if key in job and not job[key].done()
                ),
                None,
            )
            if future and not future.done():
                return
            # Keep the abandoned record until the controller acknowledges the
            # complete inventory. A lost response must not strand its lease.
            self.request("reconcile", {"running": [key for key in self.jobs if key != ident]})
            del self.jobs[ident]
            return
        if not job["container"]:
            # Download workers cannot start a container. Keep supervision responsive.
            ready = job["transfer"].done()
            # Renew even a recent lease before launching. A completed transfer
            # must not start compute after a known control-plane disconnect.
            if ready or time.monotonic() - job["renewed"] >= 5:
                self.request("heartbeat", {"token": request["token"]}, ident)
                job["renewed"] = time.monotonic()
            if not ready:
                return
            try:
                job["execution_metrics"] = job["transfer"].result()
                self.launch(request)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                # Reconcile any ambiguous creation; never assert a running container stopped.
                name = "wk-attempt-" + ident
                exists = self.docker("ps", "-aq", "--filter", "name=^/" + name + "$")
                if exists:
                    self.docker("stop", "-t", "5", exists)
                self.request(
                    "error",
                    {
                        "token": request["token"],
                        "stopped": True,
                        "category": classify(exc),
                        **self.error_details(exc, request),
                    },
                    ident,
                )
                del self.jobs[ident]
            return
        info = json.loads(self.docker("inspect", job["container"]))[0]
        if info["State"]["Running"]:
            if shutil.disk_usage(self.root).free < self.config["budgets"].get(
                "disk_headroom", 0
            ) or directory_bytes(self.root / "attempts" / ident) > request["resources"].get(
                "disk", 0
            ):
                job["abort_category"] = "disk"
                self.docker("stop", "-t", "5", job["container"])
                return
            if time.monotonic() - job["renewed"] >= 20:
                self.docker("stop", "-t", "5", job["container"])
                return
            try:
                self.request(
                    "heartbeat",
                    {
                        "token": request["token"],
                        "container_id": job["container"],
                        **(
                            json.loads(
                                (self.root / "attempts" / ident / "progress.json").read_text()
                            )
                            if (self.root / "attempts" / ident / "progress.json").exists()
                            else {}
                        ),
                    },
                    ident,
                )
                job["renewed"] = time.monotonic()
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 409}:
                    self.docker("stop", "-t", "5", job["container"])
                else:
                    raise
            return
        outcome_path = self.root / "attempts" / ident / "outcome.json"
        if info["State"]["ExitCode"] == 0 and outcome_path.exists():
            if "publication" not in job:
                job["publication"] = self.transfers.submit(self.publish_outcome, outcome_path)
            if time.monotonic() - job["renewed"] >= 5:
                self.request("heartbeat", {"token": request["token"]}, ident)
                job["renewed"] = time.monotonic()
            if not job["publication"].done():
                return
            outcome = job["publication"].result()
            self.request(
                "complete",
                {
                    "token": request["token"],
                    "stopped": True,
                    "container_finished_at": info["State"]["FinishedAt"],
                    "execution_metrics": job.get("execution_metrics", {}),
                    **outcome,
                },
                ident,
            )
        else:
            remember(self.root, request)
            if "publication" not in job and request["kind"] == "embedding":
                job["publication"] = self.transfers.submit(
                    self.publish_partial, request, outcome_path.parent
                )
            if "publication" in job:
                if time.monotonic() - job["renewed"] >= 5:
                    self.request("heartbeat", {"token": request["token"]}, ident)
                    job["renewed"] = time.monotonic()
                if not job["publication"].done():
                    return
                partial = job["publication"].result()
                if partial.get("artifacts"):
                    self.request("checkpoint", {"token": request["token"], **partial}, ident)
            error_file = outcome_path.parent / "error.json"
            category = classify(
                exit_code=info["State"]["ExitCode"],
                oom_killed=info["State"].get("OOMKilled", False),
            )
            if error_file.exists() and category == "unknown":
                category = json.loads(error_file.read_text()).get("category", "unknown")
            logs = subprocess.check_output(
                ["docker", "logs", "--tail", "200", job["container"]],
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=5,
            )[-512 * 1024 :]
            outcome_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            (outcome_path.parent / "container.log").write_text(logs)
            if category == "unknown":
                # CUDA OOM occurs in a model subprocess and need not kill the
                # cgroup. Preserve/classify its stderr instead of losing it at
                # the outer CalledProcessError boundary.
                category = classify(RuntimeError(logs))
            category = job.get("abort_category", category)
            self.request(
                "error",
                {
                    "token": request["token"],
                    "stopped": True,
                    "category": category,
                    "exit_code": info["State"]["ExitCode"],
                    "container_finished_at": info["State"]["FinishedAt"],
                    "execution_metrics": job.get("execution_metrics", {}),
                },
                ident,
            )
        del self.jobs[ident]
        self.docker("rm", job["container"])
        if info["State"]["ExitCode"] == 0 and outcome_path.exists():
            # Completion has been fenced/committed and immutable blobs are durable.
            # Retain request/outcome journals; private scratch is no longer a checkpoint.
            for folder in ("media", "tmp"):
                shutil.rmtree(outcome_path.parent / folder, ignore_errors=True)

    def publish_partial(self, request, directory):
        result = partial_manifest(request, directory)
        for artifact in result["artifacts"]:
            source = directory / safe_relative(artifact.pop("source"))
            with source.open("rb") as stream:
                self.store.publish(
                    stream,
                    artifact["checksum"],
                    artifact["size"],
                    minimum_free=self.config["budgets"]["disk_headroom"],
                )
        return result

    def publish_outcome(self, outcome_path):
        outcome = json.loads(outcome_path.read_text())
        for artifact in outcome.get("artifacts", []):
            source = (outcome_path.parent / safe_relative(artifact.pop("source"))).resolve()
            if not source.is_relative_to(outcome_path.parent):
                raise ValueError("Artifact source escapes attempt directory")
            with source.open("rb") as stream:
                self.store.publish(
                    stream,
                    artifact["checksum"],
                    artifact["size"],
                    minimum_free=self.config["budgets"]["disk_headroom"],
                )
        return outcome

    def run(self):
        from tools.executor.runtime_identity import identity

        # Startup may coincide with a controller outage. Never wait for it (or
        # checksum large models) while a previous agent's containers keep running.
        stopped = self.stop_recovered()
        runtime = identity(self.config.get("gpu_uuid", ""))
        self.config["capabilities"]["runtime_fingerprint"] = runtime["fingerprint"]
        self.config["capabilities"]["runtime"] = runtime

        if self.config["capabilities"].get("profiles") and not self.config.get("runtime_files"):
            raise ValueError("Profile activation requires pinned runtime and model files")
        stamps = self.verify_runtime_files()
        for image in self.config["images"]:
            self.retain_image(image)
        while True:
            try:
                self.request(
                    "register",
                    {"capabilities": self.config["capabilities"], "telemetry": self.observe()},
                )
                self.request("reconcile", {"running": [], "stopped": stopped})
                break
            except (OSError, ValueError, subprocess.SubprocessError):
                time.sleep(5)
        while True:
            try:
                if identity(self.config.get("gpu_uuid", ""))["fingerprint"] != runtime[
                    "fingerprint"
                ] or any(file_stamp(Path(path)) != stamp for path, stamp in stamps.items()):
                    self.config["capabilities"]["profiles"] = []
                self.tick_jobs()
                telemetry = self.observe()
                reply = self.request(
                    "register",
                    {"capabilities": self.config["capabilities"], "telemetry": telemetry},
                )
                self.reconcile_unreceived(reply)
                if time.monotonic() - self.last_cleanup >= 30:
                    self.collect_artifacts()
                for ident in reply.get("stop_attempts", []):
                    if ident in self.jobs:
                        job = self.jobs[ident]
                        if job["container"]:
                            self.docker("stop", "-t", "5", job["container"])
                        elif not job["transfer"].done():
                            job["abandoned"] = True
                        else:
                            self.request(
                                "error",
                                {
                                    "token": job["request"]["token"],
                                    "stopped": True,
                                    "category": "canceled",
                                },
                                ident,
                            )
                            del self.jobs[ident]
                if (
                    time.monotonic() - self.last_contact < 5
                    and sum(not j["container"] for j in self.jobs.values()) < 4
                ):
                    request = self.request("claim").get("attempt")
                    if request:
                        active = [
                            {
                                "resources": j["request"]["resources"],
                                "observed": telemetry["attempts"].get(k, {}),
                            }
                            for k, j in self.jobs.items()
                        ]
                        allowed = admit(
                            request["resources"],
                            active,
                            self.config["budgets"],
                            telemetry,
                            gpu=bool(request["resources"].get("vram")),
                        )
                        if allowed.fits:
                            try:
                                self.stage(request)
                            except Exception as exc:
                                # If creation was ambiguous, reconciliation must prove termination.
                                self.request(
                                    "error",
                                    {
                                        "token": request["token"],
                                        "stopped": request["id"] not in self.jobs,
                                        "category": classify(exc),
                                        **self.error_details(exc, request),
                                    },
                                    request["id"],
                                )
                        else:
                            self.request(
                                "error",
                                {
                                    "token": request["token"],
                                    "stopped": True,
                                    "category": allowed.reason,
                                },
                                request["id"],
                            )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.expire_unrenewed()
                if time.monotonic() - self.last_error_log >= 10:
                    print(
                        json.dumps(
                            {
                                "event": "executor.control_error",
                                "host": self.host,
                                **self.error_details(exc, {}),
                            }
                        ),
                        flush=True,
                    )
                    self.last_error_log = time.monotonic()
            time.sleep(1)

    def expire_unrenewed(self):
        """Fence compute and drain transfers whose ownership cannot be renewed."""
        for job in list(self.jobs.values()):
            if time.monotonic() - job["renewed"] < 20:
                continue
            if job["container"]:
                self.docker("stop", "-t", "5", job["container"])
            else:
                # Transfer threads cannot create containers. Wait for them to
                # finish, then reconcile absence before releasing reservations.
                job["abandoned"] = True

    def collect_artifacts(self):
        # The control plane excludes every pending/active checksum alias before
        # fencing deletion. Unrelated running attempts need not delay residue cleanup.
        cleanup = self.request("collect-artifacts", {"version": 2}).get("cleanup")
        if cleanup:
            # The durable cleanup record fences admissions on this host. It
            # remains pending across lost responses and executor restarts.
            atomic_json(self.root / "cleanup.json", cleanup)
            for blob in cleanup["blobs"]:
                self.store.path(blob["checksum"]).unlink(missing_ok=True)
            for ident in cleanup.get("attempt_ids", []):
                work = self.root / "attempts" / safe_relative(ident)
                if not work.resolve().is_relative_to(self.root / "attempts"):
                    raise ValueError("Cleanup attempt escapes executor storage")
                for folder in ("media", "tmp"):
                    scratch = work / folder
                    if scratch.is_symlink():
                        scratch.unlink()
                    elif scratch.exists():
                        try:
                            shutil.rmtree(scratch)
                        except FileNotFoundError:
                            pass  # A second collector may acknowledge the same durable cleanup.
            self.request("collected-artifacts", {"id": cleanup["id"], "token": cleanup["token"]})
        self.last_cleanup = time.monotonic()
        return bool(cleanup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--verify-runtime-files", action="store_true")
    parser.add_argument("--verification-threads", type=int, default=1)
    args = parser.parse_args()
    agent = Agent(json.loads(args.config.read_text()))
    if args.verify_runtime_files:
        try:
            print(
                json.dumps(
                    {"files_verified": len(agent.verify_runtime_files(args.verification_threads))}
                )
            )
        finally:
            agent.transfers.shutdown(wait=True)
    else:
        agent.run()


if __name__ == "__main__":
    main()
