# Execution operations, 12 November 2026

## Progress scans of large requests saturated production I/O (control release 142)

When GPU work resumed at about 15:40, production's I/O pressure rose to 10-14 % (`full`,
avg60 and avg300) with no attempt running there, above the 5 % at which launches are refused.
One long-lived controller connection read 400-990 MB every five seconds. It was the progress
projection of the 40,130-row LJEc7C9: `shared_embedding_summaries` counts the request's
embedding inputs produced by other requests, which joins its ~175,000 input links to the
artifact table (the planner scans the whole 2.5 GB link table and then looks up each artifact
in the 7 GB catalogue). One scan took 6-9 s and touched 1.2 M buffers. Large requests were
projected every 15 s, so the database read about half of the time; the scan also ran close
to its 10 s statement limit, and a scan that timed out was retried at the next control tick.

Reading the producers from the stage graph instead was no faster (6.2 s) and missed 11 links
without a graph edge, so the counts are unchanged and the scans are spaced instead. Commit
`65db0d6`: a large request's projection waits at least 15 s and at least nine times its last
scan (`scan_seconds` in the outbox event), so one request's progress takes at most about a
tenth of the database's time; a timed-out scan is postponed the same way. Forced refreshes
(final assembly) are still projected immediately. Control image `webkinpred-control-fix:r142`
(`sha256:6df3a33966863ccb0067dac84a7b51098cdcba148dd92b2101f6194f513c8900`) runs since
15:59:53; all 2,317 unfinished requests were present afterwards with unchanged checksums
and the legacy queue still held 5,002 messages. LJEc7C9 is now projected every ~67 s (7.4 s
scan); production's I/O pressure fell to 0.7-1.4 % (avg60) within three minutes. One
in-flight T5 residue attempt was interrupted by the restart and completed on retry.
