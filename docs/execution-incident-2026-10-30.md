# Execution stall, 30 October 2026

## Diagnosis

At 13:40 UTC, 261 jobs had completed under durable execution, but the last
completion was 29 October at 18:09:38 UTC. There were 6,018 pending jobs and 22
processing jobs. Neither executor had a running attempt container. Ten old
leases remained in the database, including eight reserving all 16 GPU-host CPU
allocations. Sample orphan IDs had no local request journals, confirming that
the executor had not received those admissions.

Recent claim requests had a median server duration of 11.6 seconds versus the
agent's five-second HTTP timeout. A timed-out claim could commit after the
client disconnected. The old agent never reconciled this unreceived reservation
while it remained running. Fresh host heartbeats therefore concealed zero
execution progress.

Backlog ranking used nested per-artifact availability queries, sorted wide
joined contexts, and paid PostgreSQL JIT compilation costs. Optional garbage
collection waited for the global preparation lock inside API workers. These
operations delayed claims and host heartbeats.

## Repair

- Registration returns the host's live reservation IDs. An executor that sees
  unknown reservations inventories its labelled containers, confirms unknown
  containers stopped, preserves their checkpoints, and reconciles. Known running
  containers and transfers keep their reservations. Lease expiry alone still
  never releases capacity.
- Admission ranks and locks lightweight candidate IDs before fetching inspected
  contexts. Availability and fairness checks avoid nested per-candidate artifact
  scans. Analytical JIT compilation is disabled within admission transactions.
- Leases start at admission after candidate selection. A bounded cold-claim
  timeout leaves time for the unchanged twenty-second local fencing deadline.
- Optional cleanup skips busy preparation/host locks and cancels expensive
  scans after 1.5 seconds, rolling back before allowing another attempt. Control errors are logged
  without ownership credentials instead of being silently swallowed.
- Regression coverage includes lost claim responses, preservation of known
  transfers, unknown-container stop confirmation, busy cleanup locks, full lease
  duration after selection, and bounded request timeouts.

Rollback-only probes on the live backlog reduced admission from 15.2 to about
1.4 seconds on the GPU host, and 11.4 to about 1.8 seconds on production when both
hosts were reachable. These are control request measurements, not model speedups.

## GPU service operation

The GPU host uses Docker containers `webkinpred-gpu-executor`,
`webkinpred-gpu-embedding-proxy`, and `webkinpred-gpu-artifacts`. The retired
`webkinpred-gpu-embed.service` must remain stopped and disabled. Starting it
conflicts with the managed compatibility endpoint; it does not repair scheduling.
The endpoint binds to the Tailscale address and requires authentication, so an
unauthenticated `curl http://127.0.0.1:8080/health` is not a valid health check.

Run this on the GPU host to check the authenticated service without printing its
token:

```sh
sudo docker exec webkinpred-gpu-embedding-proxy python -c 'import os,json,urllib.request; r=urllib.request.Request("http://100.78.161.61:8080/health",headers={"Authorization":"Bearer "+os.environ["GPU_EMBED_SERVICE_TOKEN"]}); print(json.dumps(json.load(urllib.request.urlopen(r,timeout=10))))'
```

Production verification uses `get_gpu_status(force_refresh=True)` from
`api.services.gpu_embed_service` and confirms new successful GPU attempts, rather
than relying only on endpoint or container liveness.

## Production recovery verification

The executor repair is commit `c118134`; the cleanup deadline is commit
`97513aa`. Both were pushed before deployment. Production uses web image `r36`
and executor `r28`; the GPU host uses executor `r28`. Existing model execution
images, environments, profiles, unrelated workloads and dirty checkouts were
preserved. The retired systemd embedding service remains stopped and disabled;
the Docker GPU endpoint is healthy from production and through the public GPU
status API.

At **30 October 2026, 14:24:39 UTC**:

- Completed jobs increased from 261 to **301**: **40 additional completions**.
- All **ten originally stranded stages completed** under new attempts. Their
  abandoned attempts are retained as stopped provenance records.
- All **6,301 pre-repair durable jobs** retain their identifiers, submission
  times, requested rows and selected methods. Previously completed jobs retain
  their completion timestamps. No queue purge or delete/recreate resubmission
  was used. Unfinished jobs have execution specifications and are unpaused.
- Quota bucket balances match their ledger entries. Both executor hosts are
  unpaused and reconciled. Reservations remain within their configured budgets.
- There are **zero expired leases, uncertain attempts or permanently failed
  jobs under durable execution**. One preparation stage held before this repair
  remains held; no additional preparation holds were introduced.
- GPU-host attempts completed protein embeddings for CatPred, EITLEM, IECata and
  KinForm, molecular KinForm embeddings, and CPU predictions for multiple methods.
  Production completed CPU predictions, similarity work and result assembly.
- Five recently completed result downloads matched the authoritative files by
  SHA-256. Public submissions remain open and the GPU status endpoint reports
  `online: true`.
- Fifteen-second observations after the final patch recorded no expired leases,
  up to three resident CPU attempts on the GPU host, CPU activity on production,
  and actual GPU embedding activity reaching **96% GPU utilization**. This is
  evidence of recovery and parallel execution, not a maximum-throughput claim.
- Recent admission requests had a median server time of about **125 ms**. A few
  requests remained slow; lost responses are now reconciled instead of retaining
  invisible reservations indefinitely. Host registrations and attempt heartbeats
  remained responsive in the measured post-patch window.
- **433 tests and three subtests passed**, including real PostgreSQL cancellation
  of a deliberately slow isolated cleanup query. The no-progress alert loaded
  successfully and Prometheus reported no rule errors.

Private, timestamped evidence is retained on production under
`/home/saleh/webkinpred-execution-staging/audit44/`, including the before snapshot,
claim profiles, `recovery-verification45.json`, and `recovery-samples45.jsonl`.
The original timing audit is deferred until the recovered platform has a useful
steady execution window.
