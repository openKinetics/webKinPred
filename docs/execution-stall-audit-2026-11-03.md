# Executor stall audit and recovery, 3 November 2026

The production inspection began at approximately 10:58 UTC. The supplied admin list contained 58 processing jobs; ten had processing ages above 48 hours, with a maximum of approximately 88.4 hours. All ten older jobs belonged to the configured bulk scheduling class.

## Incident

The GPU host was online, unpaused and advertising 84 validated profiles, but its durable `reconciled` flag was false. No leased, running or uncertain attempts remained. Both hosts had no active attempts at the initial snapshot. The GPU host reported approximately 18 GiB free VRAM, 121 GiB available RAM and 0.2 logical CPUs in use. This was a dispatch failure, not evidence that the backlog required continuous two-day computation.

The last GPU-host attempt before the stall finished at **02:23:37 UTC**. Dispatch remained blocked until an operator reconciliation at approximately 11:01 UTC, after verifying an empty running-container inventory and zero live database reservations. GPU execution then resumed; a direct reading reached 96% utilization with three platform attempt containers running.

The recovery protocol had a missing transition:

1. Reconciliation could mark an expired, known attempt uncertain and set the host's `reconciled` flag false.
2. The supervisor subsequently stopped/reported the attempt, clearing the last live ownership record.
3. Registration returned an empty owned-attempt list but omitted the host reconciliation flag.
4. The supervisor skipped reconciliation whenever it had no unknown claims. The stale false flag therefore blocked every subsequent claim indefinitely despite fresh heartbeats.

This explains the eight-hour GPU dispatch outage. It does not account for every hour of the older jobs' historical elapsed time. The bulk class also waits behind normal requests according to the configured priority policy.

## Fix and safe deployment

Source commit `e5f0224abb93a8e144b2ad253c98563652f05435` adds the reconciliation flag to registration responses. A supervisor receiving `reconciled=false` now inventories its identified containers and submits a full reconciliation even when the owned-attempt list is empty. Known transfers/running attempts remain in the inventory; unknown platform containers must be confirmed stopped before their reservations can be released. No flag is blindly reset on the basis of lease expiry or heartbeat freshness.

A new critical `ExecutionHostUnreconciled` alert covers an unpaused host blocked in this state. Prometheus had still mounted release 53's rules. Its service was recreated against release 100, retaining its metrics volume. `promtool` validated all 13 rules, and the new alert was confirmed loaded and inactive after recovery. This is a Prometheus alert rule; this rollout did not configure an external notification recipient.

The complete isolated suite passed **614 tests and three subtests**. New tests reproduce the final-expired-attempt transition, recovery with an empty ownership list, and preservation of known work. In a live controlled probe with admissions paused and no live attempts, both hosts were marked unreconciled; both updated supervisors independently inventoried and restored reconciliation. Their logs recorded successful recovery with zero unknown claims.

Running attempts were allowed to finish before supervisor replacement. Neither supervisor deployment interrupted a running attempt. Both hosts were resumed, and no model environments were rebuilt. Existing profiles, budgets, queue order and quota policy were retained.

Pinned images:

| Component | Image |
|---|---|
| Backend/controller/control worker | `sha256:05284d3055d1388214db499898c2dcbbc9b00ecf6ddcbb09594464ebaf98e3b7` |
| Production supervisor | `sha256:a432f0f1606fa5e4611ec150b27d264bc7158dae5bd26a162529fea9d50a1abc` |
| GPU-host supervisor | `sha256:36c786a56490f4fe82b20a8c788597953565846c46a25a70d79747b7f3d0dbe2` |

## Job and dependency checks

A bounded traversal of the supplied jobs' unfinished dependency chains examined 1,701 stages and 3,368 edges after recovery. It found **zero canceled-stage blockers and zero cyclic or cycle-blocked nodes**. The remaining leaf work was chiefly GPU embeddings. No scientific input files were scanned for this audit and no model-profiling workload was launched.

For three older examples, recorded active execution was approximately 5m52s, 23m45s and 30m53s, despite processing ages of 88.4h, 51.7h and 51.0h respectively. These are recorded executor intervals, including relevant shared producers, rather than a claim that all historical compute time is known. Their public execution state was `waiting_gpu`.

All **9,431 original unfinished-job IDs** remain present. All **11,676 original request checksums and input checksums** remain unchanged. At the preservation snapshot those source jobs comprised 926 completed, 56 processing and 8,449 pending. No broker queue was purged, jobs were not resubmitted or charged again, and original inputs/results were not deleted.

From 01:15 UTC to the initial 11:02 audit, 416 jobs covering 40,745 submitted rows completed. Of these, 239 jobs / 23,543 rows completed before 02:24 UTC. Only two jobs / 136 rows completed in the hour preceding the audit. The difference reflects a real throughput collapse, not merely a misleading elapsed-time display.

## Post-fix observation

A three-minute, one-second GPU-host probe recorded 180 samples with no errors:

- GPU utilization: **60.6% mean, 99% peak**.
- GPU-host CPU use: **3.10 / 20 logical CPUs mean, 8.10 peak**.
- Eleven coarser executor snapshots observed up to **two GPU allocations and four GPU-host CPU attempts**; production reached two CPU attempts.
- All **44 public status/health requests succeeded**, averaging 0.34 seconds.
- Four supplied CatPred jobs completed after recovery. Both hosts were online, reconciled and unpaused, with no held stages or uncertain attempts at the final status check.

The system is executing again, but these short measurements do **not** establish maximum sustained utilization. The GPU host retains substantial CPU headroom. Dependency readiness, model startup, dispatch overhead, validated mixture capacity and production publication still affect throughput. The immediate fault was the latched reconciliation gate; increasing worker/resource limits would not have removed it.

Private production evidence is retained under `/home/saleh/webkinpred-execution-staging/`: `audit100.json`, `audit100-hosts.json`, `dependencies100-final.json`, `preservation100.json`, `audit100-observation.json`, `capacity100.samples.jsonl`, `http100.samples.jsonl`, release/build/deployment records and `prometheus100.log`. GPU evidence includes `/var/lib/webkinpred/gpu100.samples.jsonl` and staging `gpu100-summary.json`. Job identifiers and inputs are omitted from this committed report.
