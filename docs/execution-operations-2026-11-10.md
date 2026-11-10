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

## GPU host clock

The GPU host's clock is 91 s ahead of production, and NTP is inactive there
(`timedatectl`: not synchronized). Supervisor log timestamps are therefore 91 s later
than the controller's. Leases use monotonic time on the supervisor and the controller's
clock on the controller, and finish times reported by containers are clamped to the
controller's time, so scheduling is unaffected. The host is shared and its time service
was left unchanged.
