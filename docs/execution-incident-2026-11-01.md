# Held IECata and KinForm predictions, 1 November 2026

## Causes

`LDy1dTR` contains one IECata request with a valid protein and an invalid
substrate. Its pinned execution image still returned `[]` as optional annotations
when every input was invalid. The native target adapter expected either absent
annotations or one annotation per input and indexed the empty list. Three retries
held the stage and suspended the IECata prediction profile. The source correction
already existed, but had not been applied to that separately pinned runtime.

`UbiEVIW` is a 1,000-row KinForm-H kcat/Km request. Its prediction stages had not
started: they depended on suspended CPU prediction profiles. Reproducing the
earlier failed `YdRSz01` attempts with Python fault handling localized SIGSEGV to
pandas binding-table type inference. Reading that table alone also crashed both
the C and Python parser engines. An ID such as `0e6681122986` resembles a numeric
exponent; identifiers and serialized residue-score lists must be read as text.
Explicit string types read the same table successfully. KinForm-L was affected
as well (`8reLvzL`). Changing parser engines alone was insufficient.

## Correction and deployment constraints

KinForm now reads binding tables with explicit text types and without implicit
NA conversion, preserving sequence identities, leading zeros, serialized scores,
row order and feature values. Ambiguous identities and missing columns are
rejected. The all-invalid native regression tests now cover IECata's `kcat/Km`
target as well as OmniESI, with one/two rows and bounded/unbounded unit batches.

Deploy the corrected generic adapter to the IECata execution image as well as
the KinForm image. Execution profiles pin model images separately from the web
image; restarting the backend cannot update their code. Use thin image overlays
and a separately checksum-pinned KinForm entrypoint mount. Preserve model weights,
tokenizers and existing Conda environments. Create new immutable profile IDs;
leave the old suspended profiles as evidence.

For supervisor configuration changes, pause admissions on the GPU host and wait
for both its database live-attempt inventory and labelled Docker container
inventory to empty naturally. Only then update its approved images, profile
fingerprints and checked runtime files, verify the files, and restart that
supervisor. Production execution, the web interface and submissions can stay
online. Resume the GPU host after importing and binding the validated profiles.

Recover held stages with `execution_admin retry --stage <UUID>` and bind profiles.
Do not recreate jobs, resubmit legacy Celery requests or purge Redis. Preserve
execution specifications, existing outputs, original IDs and quota reservations.
The low-priority client classification continues to apply to `UbiEVIW`.

## Verification gates

The local regression suite passes 487 tests and three subtests. The isolated
runtime validation covers KinForm-H kcat/Km, KinForm-L kcat and IECata kcat/Km:
three fresh-process and five filesystem-warm boundary runs per profile, compared
with the original CPU reference, plus the actual held input for each profile.
Preserve the resulting reports and checksums, and include process-tree RSS as
well as cgroup high-water marks when calculating reservations. Never lower an
existing reservation during this patch.

Before declaring deployment complete, verify supervisor registration and profile
compatibility, retry/binding results, input/identity preservation, the original
unfinished-job inventory, the frozen legacy Redis queue, and public job states.
An invalid substrate must yield an explicit invalid-row annotation, not a made-up
prediction. Do not describe a still-waiting job as completed.

Private reproductions, measurements, image identities and profile registries are
under `/home/msp/webkinpred-execution-staging/probes/repair54/` and adjacent
`release54-*` / `profiles54-*` files. Production retains its preservation inventory
and recovery script under `/home/saleh/webkinpred-execution-staging/`.

## Observed deployment result

All 36 isolated runtime probes passed, including the four reproduced failures.
The 32 boundary runs matched their previous CPU references. The final reservation
gate included process-tree RSS and retained or increased every old reservation.
Four new `gpu54-*` CPU profiles were imported for the two KinForm-H targets,
KinForm-L kcat and IECata kcat/Km. Old suspended profiles remain unchanged.

The GPU supervisor verified 9,883 runtime files. Both the live-attempt database
inventory and labelled container inventory were empty before its restart; zero
running attempts were stopped. A captured pre-deployment CatPred attempt completed
naturally. Production supervision and web services were not restarted, and no
model environment was rebuilt. The original dirty model trees were not modified.

All 9,901 jobs in the unfinished-job preservation inventory remained present.
The frozen legacy Redis queue retained its 5,002 messages. The retry/import path
made no quota changes and performed no broker purge. Both hosts resumed admissions;
the public health endpoint reported healthy with submissions open.

`LDy1dTR` completed all three stages. Its downloadable CSV returned HTTP 200 and
contained one row annotated `Invalid substrate (not a valid SMILES or InChI)`.
`UbiEVIW` cleared the operator-attention state and reported `waiting_gpu`, with
6 of 91 stages completed at the check. It remains in the background scheduling
class; its completion is not claimed. The other affected KinForm stages were
also requeued without recreating their jobs. Live IECata attempts were observed
running concurrently in the corrected image with enforced CPU and RAM limits.

## CatRange substrate context correction

`B5lzhWX` completed its embeddings but its prediction attempts failed three times
with `MISSING_EMBEDDING: molecular batch context mismatch`. Molecular planning
honored the canonicalization option; the generic CatRange prediction adapter
passed the original substrate text. Rebuilding otherwise valid embeddings did
not resolve that textual mismatch. Automatic recovery held the stage and suspended
its GPU-host kcat prediction profile.

Commit `45a28b0` normalizes only CatRange's subprocess substrate inputs with the
same `substrate_as_smiles` helper and options used by molecular planning. Raw
SMILES remain raw when requested; InChI conversion and stereochemistry follow
the same policy on both paths. Invalid rows retain their annotations and original
positions. Original uploaded files and other methods' encoder contexts are unchanged.

Regression coverage includes both targets, raw/canonical options, InChI, invalid
rows, row order and one-unit retry batches. The full suite passed 495 tests and
three subtests. Both final deployment runtimes passed three fresh-process and five
filesystem-warm boundary runs per target against the earlier CPU references, plus
the saved ten-row failed input per target: 36 isolated runtime checks in total.
The saved-input result digests matched exactly between the two hosts for each
target. Resource envelopes include process-tree RSS and never reduce old limits.

Thin overlays replace the generic adapter in the separately pinned production
and GPU-host CatRange execution images; no environment or weight was rebuilt.
The four new `production55-*` / `gpu55-*` profiles replace the affected version-29
profiles, which remain suspended. Supervisors were restarted only after live
database attempts and labelled containers had drained; zero running attempts
were stopped. Production verified 926 runtime files and the GPU host verified
9,883. The web services and submission interface stayed online.

Only the held prediction stage was retried. Its live attempt completed under
`gpu55-CatRange-kcat`. All 9,731 jobs from the preservation snapshot remained
present, including the original execution specification and input checksum for
`B5lzhWX`; the frozen Redis queue retained 5,002 messages. The three attempts
captured before deployment all completed naturally. Both hosts resumed admissions,
all four profiles remained validated and enabled, and no new-profile stage was
held at the verification check.

After assembly, similarity and publication, `B5lzhWX` completed all 18 stages.
Its public download returned HTTP 200 (4,646 bytes). The CSV contained the original
10 rows with 10 nonempty kcat predictions, each sourced from CatRange. The public
status endpoint reported `Completed` / `completed`. No model environment was
rebuilt, no running attempt was killed during rollout, and no queue was purged.

Private deployment, calibration and preservation evidence is under the two hosts'
`webkinpred-execution-staging/` directories in `release55-*`, `profiles55-*`,
`preservation55.json` and `probes/repair55/`. Copied job reproductions remain private.
