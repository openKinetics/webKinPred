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
