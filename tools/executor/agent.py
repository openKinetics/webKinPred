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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import psutil
from api.execution.artifacts import ArtifactStore, atomic_json, safe_relative, sha256_file
from api.execution.errors import classify
from api.execution.resources import admit
from api.execution.telemetry import container_observations, snapshot

from tools.executor.checkpoints import partial_manifest, remember

LABEL = "org.webkinpred.executor"
# Admission throughput is bounded by how many claimed attempts may transfer
# inputs at once and how many attempts one claim request may return.
TRANSFER_WORKERS = 8
PENDING_TRANSFERS = 8
CLAIM_BATCH = 8
# The host GPU identity and runtime files are stable; checking them on every
# one-second supervision tick spent most of the tick in nvidia-smi and stat().
RUNTIME_CHECK_SECONDS = 30
LEGACY_CHECK_SECONDS = 30
DISK_CHECK_SECONDS = 5
EXTERNAL_WINDOW_SECONDS = 20


class LeaseLost(Exception):
    """A batched renewal was rejected exactly as a single heartbeat would be."""

    def __init__(self, code):
        super().__init__(f"Lease renewal rejected with HTTP {code}")
        self.code = code


class RuntimeMismatch(ValueError):
    """A pinned runtime or model file no longer has its recorded digest."""


def publishes_only(request):
    """The final publication phase computes nothing; its runner only hashes the result."""
    return (
        request.get("kind") == "assemble" and request.get("payload", {}).get("phase") == "publish"
    )


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
        self.local_budgets = deepcopy(config.get("budgets", {}))
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
        self.next_claim_at = 0.0
        self.empty_claim_delay = 1.0
        self.claim_owners = frozenset()
        self.claim_paused = False
        self.transfers = ThreadPoolExecutor(
            max_workers=TRANSFER_WORKERS, thread_name_prefix="artifact-transfer"
        )
        # Completion commits can take seconds for large manifests. They are
        # idempotent, so they run beside supervision instead of stalling it.
        self.control = ThreadPoolExecutor(max_workers=4, thread_name_prefix="control-commit")
        # Re-hashing a large model must not stall lease renewal.
        self.verifier = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-verify")
        self.batch_heartbeats = True
        self.batch_claims = True
        self.observed_at = -math.inf
        self.observation = None
        self.legacy_checked_at = -math.inf
        self.legacy_consumers = []
        self.inspected = {}
        self.timings, self.counts, self.iterations = {}, {}, 0
        self.timings_since = time.monotonic()
        self.external_samples = deque()
        self.runtime, self.runtime_stamps = None, {}
        self.runtime_state, self.runtime_checked, self.reverification = "verified", 0.0, None

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
                raise RuntimeMismatch("Runtime/model checksum mismatch; executor remains offline")
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

    def runtime_verified(self):
        """Whether profiles measured on this runtime may still be advertised.

        A different GPU identity, or a pinned file whose bytes no longer match
        its digest, stops advertising until restart. A new stamp alone is not
        a change: huggingface_hub rewrites ``refs/<revision>`` with the same
        commit whenever any process on the host resolves that model online.
        Such files are re-hashed beside supervision; nothing is advertised
        until their digests are confirmed.
        """
        from tools.executor.runtime_identity import identity

        if self.runtime_state == "changed":
            return False
        if self.reverification is not None:
            if not self.reverification.done():
                return False
            pending, self.reverification = self.reverification, None
            try:
                self.runtime_stamps = pending.result()
                self.set_runtime_state("verified")
            except RuntimeMismatch as exc:
                self.set_runtime_state("changed", reason=str(exc))
                return False
            except (OSError, ValueError) as exc:
                self.set_runtime_state("unverified", reason=str(exc))
        if time.monotonic() - self.runtime_checked < RUNTIME_CHECK_SECONDS:
            return self.runtime_state == "verified"
        self.runtime_checked = time.monotonic()
        try:
            if (
                identity(self.config.get("gpu_uuid", ""))["fingerprint"]
                != self.runtime["fingerprint"]
            ):
                self.set_runtime_state("changed", reason="GPU runtime identity changed")
                return False
            drifted = [
                path
                for path, stamp in self.runtime_stamps.items()
                if file_stamp(Path(path)) != stamp
            ]
        except (OSError, subprocess.SubprocessError) as exc:
            self.set_runtime_state("unverified", reason=str(exc))
            return False
        if drifted:
            self.set_runtime_state("unverified", reason="stamp changed", files=drifted[:5])
            self.reverification = self.verifier.submit(self.verify_runtime_files)
            return False
        self.set_runtime_state("verified")
        return True

    def set_runtime_state(self, state, **details):
        if state != self.runtime_state:
            print(
                json.dumps({"event": "executor.runtime_" + state, "host": self.host, **details}),
                flush=True,
            )
        self.runtime_state = state

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

    def inspect_many(self, containers):
        """Inspect every supervised container with one CLI call, keyed by ID.

        A container missing from the reply is left out; callers then inspect it
        individually, which raises exactly as before.
        """
        if not containers:
            return {}
        result = subprocess.run(
            ["docker", "inspect", *containers],
            capture_output=True,
            text=True,
            timeout=20,
        )
        try:
            rows = json.loads(result.stdout or "[]")
        except ValueError:
            return {}
        return {row["Id"]: row for row in rows if isinstance(row, dict) and "Id" in row}

    def compute_finished(self, job):
        """True once the container exited (or none was needed) and only publication remains."""
        if job.get("inline") or "publication" in job or "completion" in job:
            return True
        info = self.inspected.get(job["container"]) if job["container"] else None
        if not info or info["State"]["Running"]:
            return False
        # A created-but-not-started container is not finished.
        return info["State"].get("Status", "exited") in {"exited", "dead"}

    def container_info(self, job):
        if job.get("inline"):
            return job["inline"]
        info = self.inspected.get(job["container"])
        if info is None:
            info = json.loads(self.docker("inspect", job["container"]))[0]
        return info

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
        # Reconciliation can flag an expired known attempt as uncertain. Its
        # subsequent stop/error clears the attempt but does not itself reconcile
        # the host. An empty owned list must still trigger a full inventory pass
        # when the controller says this host needs reconciliation.
        if not unknown and reply.get("reconciled", True):
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
        from tools.executor.transfers import materialize

        return materialize(self, request, work)

    def refresh_capabilities(self):
        """Reload only measured profiles/mixtures, never execution images or mounts."""
        path = self.config.get("capabilities_file")
        if not path:
            return
        path = Path(path)
        if path.stat().st_size > 1024**2:
            raise ValueError("Capability registry exceeds limit")
        from api.execution.artifacts import HEX

        registry = json.loads(path.read_text())
        profiles = registry["profiles"]
        combinations = registry["validated_gpu_combinations"]
        if (
            not isinstance(profiles, list)
            or not isinstance(combinations, list)
            or any(not isinstance(p, str) or not HEX.fullmatch(p) for p in profiles)
        ):
            raise ValueError("Invalid capability profiles")
        allowed = set(profiles)
        if any(
            not isinstance(c, list)
            or not 2 <= len(c) <= 8
            or any(p not in allowed for p in c)
            or sorted(c) != c
            for c in combinations
        ):
            raise ValueError("Invalid capability combinations")
        self.config["capabilities"].update(
            profiles=profiles, validated_gpu_combinations=combinations
        )

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

    def run_inline(self, ident, job):
        """Produce the publication outcome that ``runner.main`` writes, without a container.

        This phase reads no model, image or environment: it verifies and hashes
        the similarity-enriched result already copied into private attempt media.
        Starting a container for it cost several seconds of scarce production CPU
        for every completed job.
        """
        request = job["request"]
        work = self.root / "attempts" / ident
        outputs = []
        for output in request["payload"].get("outputs", []):
            path = work / "media" / safe_relative(output["path"])
            if not path.is_file():
                raise ValueError("Execution did not produce every declared output")
            outputs.append(
                {
                    "id": output["id"],
                    "identity": output["identity"],
                    "checksum": sha256_file(path),
                    "size": path.stat().st_size,
                    "metadata": {},
                    "source": "media/" + output["path"],
                }
            )
        atomic_json(
            work / "outcome.json",
            {
                "artifacts": outputs,
                "checkpoint": request["payload"].get("assembly_checkpoint", {}),
            },
        )
        job["inline"] = {
            "State": {
                "Running": False,
                "ExitCode": 0,
                "OOMKilled": False,
                "FinishedAt": datetime.now(timezone.utc).isoformat(),
            }
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
        telemetry = self.recent_observation(2.0) or self.observe()
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

    def recent_observation(self, max_age):
        """Let back-to-back launches share one fresh host measurement.

        Reuse is safe for admission because every newer attempt is still counted
        through its fully unspent reservation, not through stale measurements.
        """
        if self.observation is not None and time.monotonic() - self.observed_at <= max_age:
            return deepcopy(self.observation)
        return None

    def observe(self):
        data = snapshot(self.root, self.config.get("gpu_uuid", ""))
        data["attempts"] = container_observations(
            {key: job["container"] for key, job in self.jobs.items() if job["container"]}
        )
        for ident, job in self.jobs.items():
            if self.compute_finished(job):
                # Its processes, memory and GPU contexts are gone; admission keeps
                # only its disk reservation while outputs publish and commit.
                data["attempts"][ident] = {"released": True}
        raw_external = max(
            0, data["cpu_used"] - sum(value.get("cpu", 0) for value in data["attempts"].values())
        )
        # A container has no CPU delta on its first observation and an exited
        # one's final interval is unattributed, so attempt churn briefly looks
        # like other host load. The persistent floor over a short window is the
        # real external use; a sustained external rise still shows within it.
        now = time.monotonic()
        self.external_samples.append((now, raw_external))
        while now - self.external_samples[0][0] > EXTERNAL_WINDOW_SECONDS:
            self.external_samples.popleft()
        data["cpu_external"] = min(value for _, value in self.external_samples)
        data["cpu_external_raw"] = raw_external
        if time.monotonic() - self.legacy_checked_at >= LEGACY_CHECK_SECONDS:
            rows = self.docker(
                "ps",
                "--filter",
                "label=com.docker.compose.project="
                + self.config.get("legacy_project", "webkinpred"),
                "--format",
                '[{{json (.Label "com.docker.compose.service")}},'
                '{{json (.Label "org.webkinpred.profiling")}},'
                '{{json (.Label "org.webkinpred.executor")}}]',
            ).splitlines()
            # docker-committed runtime images retain their old Compose labels.
            # Identify our isolated probes/attempts before fencing legacy consumers.
            self.legacy_consumers = [
                service
                for service, profiling, executor in map(json.loads, rows)
                if service in {"celery", "celery-cache", "celery-beat"}
                and profiling != "true"
                and not executor
            ]
            self.legacy_checked_at = time.monotonic()
        data["legacy_consumers"] = list(self.legacy_consumers)
        self.observation, self.observed_at = deepcopy(data), time.monotonic()
        return data

    def renew_leases(self):
        """Renew every due lease with one request instead of one request per attempt.

        Attempts about to launch are renewed too, so compute never starts after
        a known control-plane disconnect. Rejections are handled per attempt.
        """
        if not self.batch_heartbeats:
            return
        now = time.monotonic()
        entries, launching = [], set()
        for ident, job in self.jobs.items():
            # A submitted completion is either in flight or about to be finalized
            # by this tick; renewing a committed attempt would be rejected and
            # divert it from ordinary cleanup. A server error proves nothing was
            # committed, so that lease is renewed again before resubmission; a
            # lost reply may hide a commit, so that one is resubmitted unrenewed.
            if job.get("abandoned") or "completion" in job or job.get("commit_unknown"):
                continue
            if (
                not job["container"]
                and not job.get("inline")
                and "transfer" in job
                and job["transfer"].done()
            ):
                launching.add(ident)
            elif now - job["renewed"] < 4:
                continue
            entry = {}
            if job["container"]:
                progress = self.root / "attempts" / ident / "progress.json"
                if progress.exists():
                    entry.update(json.loads(progress.read_text()))
                entry["container_id"] = job["container"]
            entry.update(id=ident, token=job["request"]["token"])
            entries.append(entry)
        if not entries:
            return
        try:
            reply = self.request("heartbeats", {"attempts": entries})
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self.batch_heartbeats = False  # An older controller: renew individually.
            return
        renewed = time.monotonic()
        for ident, result in reply.get("results", {}).items():
            job = self.jobs.get(ident)
            if job is None:
                continue
            if "lease_seconds" in result:
                job["renewed"] = renewed
                if ident in launching:
                    # Renewed strictly after its transfer finished.
                    job["launch_renewed"] = True
            elif result.get("status") in {403, 409}:
                job["lease_error"] = result["status"]

    def lease_lost(self, ident, job):
        # A stale result is never committed. Reconcile only after proving
        # this container stopped, retaining every other live reservation.
        future = next(
            (
                job[key]
                for key in ("transfer", "publication", "completion")
                if key in job and not job[key].done()
            ),
            None,
        )
        self.count("lease_lost")
        if future and not future.done():
            job["abandoned"] = True
            return
        if job["container"]:
            info = json.loads(self.docker("inspect", job["container"]))[0]
            if info["State"]["Running"]:
                self.docker("stop", "-t", "5", job["container"])
            # Nothing supervises this container again; journals and outputs
            # live in the attempt directory. Leaving it accumulated hundreds.
            subprocess.run(["docker", "rm", job["container"]], capture_output=True, timeout=20)
        del self.jobs[ident]
        self.request("reconcile", {"running": list(self.jobs)})

    def tick_jobs(self):
        self.inspected = self.inspect_many(
            [job["container"] for job in self.jobs.values() if job["container"]]
        )
        self.renew_leases()
        for ident, job in list(self.jobs.items()):
            try:
                if job.get("lease_error") and not job.get("abandoned"):
                    code = job.pop("lease_error")
                    info = self.inspected.get(job["container"]) if job["container"] else None
                    if info and info["State"]["Running"]:
                        # As for a rejected single heartbeat: fence compute now
                        # and let the exit path preserve checkpoints.
                        self.docker("stop", "-t", "5", job["container"])
                        continue
                    raise LeaseLost(code)
                self.tick_job(ident, job)
            except (urllib.error.HTTPError, LeaseLost) as exc:
                if exc.code not in {403, 409}:
                    raise
                self.lease_lost(ident, job)
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
                    for key in ("transfer", "publication", "completion")
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
        if not job["container"] and not job.get("inline"):
            # Download workers cannot start a container. Keep supervision responsive.
            ready = job["transfer"].done()
            # Renew even a recent lease before launching. A completed transfer
            # must not start compute after a known control-plane disconnect.
            # The batched renewal normally did this moments ago.
            renewed_for_launch = ready and job.pop("launch_renewed", False)
            if (ready and not renewed_for_launch) or time.monotonic() - job["renewed"] >= 5:
                self.request("heartbeat", {"token": request["token"]}, ident)
                job["renewed"] = time.monotonic()
            if not ready:
                return
            try:
                job["execution_metrics"] = job["transfer"].result()
                if publishes_only(request):
                    self.run_inline(ident, job)
                    self.count("published_inline")
                else:
                    self.launch(request)
                    self.count("launched")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.count("launch_failed")
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
                if not exists:
                    # No container ever ran: the private copies are only
                    # materialized inputs (blobs and checkpoints stay in their
                    # own stores). Keeping them held gigabytes per refused
                    # attempt until its stage eventually completed.
                    for folder in ("media", "tmp"):
                        shutil.rmtree(self.root / "attempts" / ident / folder, ignore_errors=True)
            return
        info = self.container_info(job)
        if info["State"]["Running"]:
            if shutil.disk_usage(self.root).free < self.config["budgets"].get("disk_headroom", 0):
                job["abort_category"] = "disk"
                self.docker("stop", "-t", "5", job["container"])
                return
            # Walking large private input trees every tick dominated supervision
            # for thousand-file prediction chunks; the host free-space guard above
            # still runs on every tick.
            if time.monotonic() - job.get("disk_checked", -math.inf) >= DISK_CHECK_SECONDS:
                job["disk_checked"] = time.monotonic()
                if directory_bytes(self.root / "attempts" / ident) > request["resources"].get(
                    "disk", 0
                ):
                    job["abort_category"] = "disk"
                    self.docker("stop", "-t", "5", job["container"])
                    return
            if time.monotonic() - job["renewed"] >= 20:
                self.docker("stop", "-t", "5", job["container"])
                return
            if time.monotonic() - job["renewed"] < 5:
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
            # Once a completion is submitted the attempt may already be committed;
            # renewing it would be rejected and divert it from ordinary cleanup.
            if (
                time.monotonic() - job["renewed"] >= 5
                and "completion" not in job
                and not job.get("commit_unknown")
            ):
                self.request("heartbeat", {"token": request["token"]}, ident)
                job["renewed"] = time.monotonic()
            if not job["publication"].done():
                return
            if "completion" not in job:
                outcome = job["publication"].result()
                job["completion"] = self.control.submit(
                    self.request,
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
                return
            if not job["completion"].done():
                return
            try:
                job["completion"].result()
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 409}:
                    raise  # The lease is gone: reconcile instead of retrying.
                # A server error (e.g. an aborted deadlock victim) is transient.
                # Re-raising the cached failure every tick would stop all
                # supervision; renew and submit the idempotent commit again.
                del job["completion"]
                return
            except OSError:
                # A lost reply may hide a committed result. A renewal would then
                # be rejected and divert the attempt from ordinary cleanup, so
                # submit the identical, idempotent commit again without one.
                del job["completion"]
                job["commit_unknown"] = True
                return
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
            logs = (
                ""
                if job.get("inline")
                else subprocess.check_output(
                    ["docker", "logs", "--tail", "200", job["container"]],
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    timeout=5,
                )[-512 * 1024 :]
            )
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
        if not job.get("inline"):
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

    def apply_control_budgets(self, reply):
        """A calibration reservation also applies at the final local launch check."""
        server = reply.get("budgets", {})
        self.config["budgets"] = {
            key: (max(value, server[key]) if key.endswith("headroom") else min(value, server[key]))
            if key in server
            else value
            for key, value in self.local_budgets.items()
        }
        for key, value in server.items():
            if key.endswith("headroom") and key not in self.config["budgets"]:
                self.config["budgets"][key] = value

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
        self.runtime, self.runtime_stamps = runtime, stamps
        self.runtime_checked = time.monotonic()
        while True:
            try:
                self.refresh_capabilities()
                capabilities = self.config["capabilities"]
                if not self.runtime_verified():
                    # Never advertise profiles measured on a different or
                    # unconfirmed runtime. The configured list is kept so a
                    # confirmed rewrite restores it without a restart.
                    capabilities = {**capabilities, "profiles": []}
                with self.timed("tick_jobs"):
                    self.tick_jobs()
                with self.timed("observe"):
                    telemetry = self.observe()
                with self.timed("register"):
                    reply = self.request(
                        "register", {"capabilities": capabilities, "telemetry": telemetry}
                    )
                self.apply_control_budgets(reply)
                self.reconcile_unreceived(reply)
                if time.monotonic() - self.last_cleanup >= 30:
                    with self.timed("collect"):
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
                with self.timed("claim"):
                    self.claim_available(telemetry, paused=reply.get("paused", False))
                self.iterations += 1
                self.report_timings()
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

    @contextmanager
    def timed(self, phase):
        began = time.monotonic()
        try:
            yield
        finally:
            self.timings[phase] = self.timings.get(phase, 0.0) + time.monotonic() - began

    def report_timings(self):
        """Emit one bounded supervision summary per minute for admission tuning."""
        now = time.monotonic()
        if now - self.timings_since < 60 or not self.iterations:
            return
        print(
            json.dumps(
                {
                    "event": "executor.loop",
                    "host": self.host,
                    "iterations": self.iterations,
                    "seconds": round(now - self.timings_since, 1),
                    "jobs": len(self.jobs),
                    "runtime": self.runtime_state,
                    "mean_phase_seconds": {
                        key: round(value / self.iterations, 3)
                        for key, value in sorted(self.timings.items())
                    },
                    **self.counts,
                }
            ),
            flush=True,
        )
        self.timings, self.counts, self.iterations, self.timings_since = {}, {}, 0, now

    def count(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0) + amount

    def claim_requests(self, limit):
        """Ask for up to ``limit`` attempts; an older controller returns at most one."""
        self.count("claims")
        if not self.batch_claims:
            attempt = self.request("claim").get("attempt")
            attempts = [attempt] if attempt else []
        else:
            reply = self.request("claim", {"limit": limit})
            if "attempts" not in reply:
                self.batch_claims = False
                attempts = [reply["attempt"]] if reply.get("attempt") else []
            else:
                attempts = [attempt for attempt in reply["attempts"] if attempt]
        self.count("claimed", len(attempts))
        return attempts

    def claim_available(self, telemetry, *, paused=False):
        """Fill capacity without letting admissions delay lease supervision."""
        began = time.monotonic()
        owners = frozenset(self.jobs)
        if owners != self.claim_owners or (self.claim_paused and not paused):
            self.next_claim_at = 0.0
            self.empty_claim_delay = 1.0
        self.claim_owners, self.claim_paused = owners, paused
        if began < self.next_claim_at:
            return
        for _ in range(4):
            now = time.monotonic()
            pending = sum(
                not job["container"] and not job.get("inline") for job in self.jobs.values()
            )
            if (
                paused
                or now - self.last_contact >= 5
                or now - began >= 4
                or any(
                    now - job["renewed"] >= 5
                    for job in self.jobs.values()
                    if "completion" not in job
                )
                or pending >= PENDING_TRANSFERS
            ):
                break
            # Continue polling for foreground preemption, but repeated empty
            # responses need not rank the whole graph every second. The delay
            # is bounded at five seconds; supervision remains independent.
            limit = min(CLAIM_BATCH, PENDING_TRANSFERS - pending)
            requests = self.claim_requests(limit)
            if not requests:
                self.next_claim_at = time.monotonic() + self.empty_claim_delay
                self.empty_claim_delay = min(5.0, self.empty_claim_delay * 2)
                break
            self.next_claim_at = 0.0
            self.empty_claim_delay = 1.0
            # Every attempt in the reply already holds a controller reservation:
            # each one is either staged here or explicitly returned to waiting.
            refused = False
            for request in requests:
                active = [
                    {
                        "resources": job["request"]["resources"],
                        "observed": telemetry["attempts"].get(ident, {}),
                    }
                    for ident, job in self.jobs.items()
                ]
                allowed = admit(
                    request["resources"],
                    active,
                    self.config["budgets"],
                    telemetry,
                    gpu=bool(request["resources"].get("vram")),
                )
                if not allowed.fits:
                    refused = True
                    self.count("refused_locally")
                    self.request(
                        "error",
                        {"token": request["token"], "stopped": True, "category": allowed.reason},
                        request["id"],
                    )
                    continue
                try:
                    self.stage(request)
                except Exception as exc:
                    refused = True
                    # Ambiguous creation retains ownership until reconciliation.
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
            if refused or (self.batch_claims and len(requests) < limit):
                break  # Refresh telemetry before asking for another allocation.

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
