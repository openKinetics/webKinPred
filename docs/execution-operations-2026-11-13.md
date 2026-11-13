# Execution operations, 13 November 2026

## GPU supervisor lost every lease after slow ticks (supervisor release 143)

From about 10:40 the GPU supervisor lost the leases of all its running attempts at once,
three times in 40 minutes (6, 6 and 7 attempts, reported as `executor_reconciled`); their
stages reran. Each loss followed a loop window whose last iteration took 20-40 s, almost all
in `tick_jobs`, with 13-15 launches in the minute (single-protein ProtT5 embeddings of about
30 s each, after their cohort profiles were suspended by shared-GPU OOMs at 02:41). Launches
and completions run one job after another in `tick_jobs`, and the batched lease renewal only
ran at its start, so a long tick let the 30 s control lease expire for every other attempt.
Production answered the supervisor quickly throughout (heartbeats p50 115 ms, max 0.9 s);
docker create/start took about 0.5 s.

Commit `bdcc026` renews leases again whenever 8 s have passed within a tick
(`RENEWAL_SECONDS`) and adds `max_phase_seconds` to the per-minute `executor.loop` summary, so
the slowest iteration of each phase is visible beside its mean. Image
`webkinpred-supervisor:r143-gpu` (`sha256:c0ced8a49c993d590d3279bf6cb448699adac2e3b247c59b5475c6ba920748f5`,
overlay on `r138-gpu`) replaced the GPU supervisor at 11:25:57 after a drain (about 20 s
without admissions, no attempt running at replacement; the previous container is kept as
`webkinpred-gpu-executor-rollback143`). Afterwards the loop ran 25-29 iterations per minute
(7-16 before) with the slowest `tick_jobs` at 1.2-1.6 s. The production supervisor still runs
release 138.
