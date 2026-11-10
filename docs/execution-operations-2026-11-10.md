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

## Large requests held their own chunks back (release 132)

Four normal 2,379-row requests and a 1,269-row request arrived from 09:01. The
controller binds resource profiles only after a preparation returns and does nothing
else meanwhile; each large request takes about eight minutes to plan, so stages of
committed chunks waited unbound (98 since 09:08, 184 since 09:12), and the 6-row
request submitted at 09:07 waited behind them. Two preparations also failed with
`Artifact cleanup is in progress` when a production deletion was pending, which holds
the request for a minute and replans its committed chunks on resume (27Ri6Pe after 13
chunks). Commit `1bd58c8` binds after every committed chunk and waits up to two minutes
for a pending deletion before publishing a chunk (the deletion fence is unchanged).
After the deploy no stage stayed unbound.

### Image depth limit

The overlay chain had reached 127 layers: the first r132 build (one more `COPY`) failed
to start with `max depth exceeded`. Above `webkinpred-execution-web:r19` (19 layers, a
layer prefix of r131) every layer only copied code into `/app` or set metadata, and the
environment, command, working directory and health check were identical. Release 132
is therefore `FROM webkinpred-execution-web:r19` plus one layer holding the 73 code files
that differ between r19 and r131 (taken from r131) and the r132 overlay: 21 layers. Its
code checksums equal r131's except `planning.py`, and its `models/` and `fastas/` file
lists (sizes and times) equal r131's. Build later releases on r132.

Control image `webkinpred-control-fix:r132`
(`sha256:bba5d4f64e463211a22e0abe575c12f3f6d462cf08977100516d4487ed7d04dd`) runs since
09:29; environment backups `live20/*.before132`. All 2,007 unfinished jobs were present
afterwards with unchanged checksums; the legacy Redis queue still held 5,002 messages.

### Release 133

Waiting until no deletion was pending still lost the race: production deletions are
claimed and finished within the same second, and at 09:32 one was claimed between the
wait and the publication, holding nfAfUCe. Commit `a85000e` retries the chunk's first
publication instead (up to two minutes). The refusal rolls back publication's savepoint,
which releases the graph lock (verified on PostgreSQL 17), and nothing else in the
chunk has run at that point. Control image `webkinpred-control-fix:r133`
(`sha256:f11a5d88b218ba6455f6b86e0932cb618d40350a541889af06232e1adc956a7b`, 22 layers)
runs since 09:57; all 2,005 unfinished jobs were present afterwards with unchanged
checksums and the legacy queue still held 5,002 messages.

## Recovered embeddings ran one tiny batch at a time (release 134)

From 09:50 the GPU averaged 8 % busy (48 of 60 one-second samples idle) while 5,400
embedding stages were ready, and completions fell to 1–14 jobs per 30 minutes. The
five normal requests reuse proteins embedded by earlier jobs, whose residue-level
outputs had been deleted after their consumers finished (only pooled vectors are kept).
Recovery regenerates them with one stage per lost producer holding only the lost
sequences: 1,932 stages of 1–5 sequences, each a 17–20 s attempt dominated by model
start-up. Cohort planning could not merge them: `recovery_origin`/`recovery_group`
differ per lost producer, and copies of an enlarged producer keep that cohort profile,
which was never a merge source.

Commit `7338ce1` lets recovery copies (`restore-batch:` keys, selected without reading
payloads) merge under their enlarged profile and ignores recovery bookkeeping when
comparing otherwise identical work; the merged unit is its own recovery origin.
Sequences are still embedded one at a time with the same model and settings. Within
four minutes of the deploy 1,484 recovery stages had merged into 193 units and 377
remained; GPU attempts carried 10–128 sequences instead of 1–3. Control image
`webkinpred-control-fix:r134`
(`sha256:2c50d88d7b824ba351490217126d4fdd90049fcf57451fb13f6b4b13a8ac28e6`, 23 layers)
runs since 10:30; all 2,004 unfinished jobs were present afterwards with unchanged
checksums and the legacy queue still held 5,002 messages.

## GPU host clock

The GPU host's clock is 91 s ahead of production, and NTP is inactive there
(`timedatectl`: not synchronized). Supervisor log timestamps are therefore 91 s later
than the controller's. Leases use monotonic time on the supervisor and the controller's
clock on the controller, and finish times reported by containers are clamped to the
controller's time, so scheduling is unaffected. The host is shared and its time service
was left unchanged.
