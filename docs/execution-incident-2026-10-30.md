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
