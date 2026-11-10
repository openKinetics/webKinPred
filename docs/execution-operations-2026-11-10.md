# Execution operations, 10 November 2026

## Claims stopped as `executor_reconciled` (release 130)

From 00:45 to 01:15, 15 GPU attempts were stopped as `executor_reconciled` within a
few seconds of being claimed, 7 of them from one claim at 01:05. Each followed an
`executor.control_error` TimeoutError and an `executor.claim_response_recovered` in the
GPU supervisor: the controller committed the claim (2.2 s for the 01:05 claim), the
supervisor had already given up waiting, and the next register stopped the
attempts it had never received.

The supervisor shortens a claim's wait so that a slow admission cannot outlast local
fencing of running containers (`15 - oldest renewal`, between 1 and 8 s). The oldest
renewal was taken over every job, including attempts whose container had finished and
whose commit was in flight, abandoned attempts and attempts with an unknown commit.
Heartbeats never renew those, so a single slow completion cut the wait to one second.
Only leases that heartbeats renew now count (commit `047263f`); claims start only when
those were renewed within 5 s, so the wait is normally the full 8 s.

| Supervisor | Image | Rollback container |
|---|---|---|
| GPU | `webkinpred-supervisor:r130-gpu` `sha256:76886e9313681cffd6d4e00f2b569b0ab10fcfc32d9c1f8f36478ab75b5f2dbe` | `webkinpred-gpu-executor-rollback130` |
| Production | `webkinpred-supervisor:r130-production` `sha256:3ad2165fef46b3b0b088e08c1d9bd2d32cbdef6ec02ebd2284746039c1fc7ace` | `webkinpred-execution-agent-1-rollback130` |

The first GPU roll (01:29–01:33) failed after draining: the copied
`replace_agent.py` still named its rollback container `…-rollback125`, which already
existed, so the rename failed after the old supervisor had stopped. The release-125
supervisor was restarted at 01:33 and the corrected roll completed at 01:35. No
attempt was lost; the GPU claimed nothing for about six minutes.

## Production claims fail while the GPU host is away (release 131)

From 01:30 to 01:33, while the GPU host had no supervisor, 41 production claims
returned 503 after the 3.2 s statement limit. Once any host's heartbeat is more than
10 s old, claims rank stages whose inputs exist only on unreachable hosts after
reachable ones. That set was derived from every ready location on the offline host
(3.8 million on the GPU host, 2.9 million of them without a production copy) using
`NOT IN`, which PostgreSQL cannot plan as an anti-join; the expression alone ran past
30 s. Every production claim failed for as long as the GPU host was away (it went
offline repeatedly on 9 November), including claims for work whose inputs production
held.

Commit `d8642fc` resolves stranded inputs from the runnable stages' own dependencies
(0.57 s on production with the GPU host treated as offline; 3,098 runnable stages had
109 artifact inputs) and passes those stage IDs to the same ordering, foreground demand
check and preemption/drain guards. With every host online nothing extra runs.

Control image `webkinpred-control-fix:r131`
(`sha256:0d3d10ac72b5136f1b3c74dd5b31dfd3bb1ac9009b5d0ea1f554d967eeff996d`, built on r129,
so it also carries migration 0045, still unapplied) runs the backend, controller and
control worker since 02:26. Environment backups `live20/*.before131`. All 3,497
unfinished jobs were present afterwards with unchanged checksums, and the legacy
Redis queue still held 5,002 messages.

## GPU host clock

The GPU host's clock is 91 s ahead of production, and NTP is inactive there
(`timedatectl`: not synchronized). Supervisor log timestamps are therefore 91 s later
than the controller's. Leases use monotonic time on the supervisor and the controller's
clock on the controller, and finish times reported by containers are clamped to the
controller's time, so scheduling is unaffected. The host is shared and its time service
was left unchanged.
