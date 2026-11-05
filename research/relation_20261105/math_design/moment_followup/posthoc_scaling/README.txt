POST-HOC COMPUTATIONAL SCALING EXTENSION: SOURCE PREPARATION ONLY

Reason and boundary
The completed first twelve synthetic cases showed no model-policy regret, but
the moment implementation was slower than strong streamed exact in all twelve
one-shot timings. The H128 compressed cases used fewer principal working arrays;
near-one cases fell back to exact. Those results remain consumed and unchanged.
This later extension asks whether H-squared versus H-times-D arithmetic produces
a crossover at larger H. It is exploratory post-hoc computational characterization,
not independent confirmation, a material score improvement, or a new algorithm.

Fixed scope
Run every H in {256,512,1024,2048} crossed with rmax {.25,.6,.9,.99}: sixteen new
cases in the declared order. Reuse exactly run_moment_control.run_case from the
two already-frozen fa2e63.../d55f39... files. Do not execute H8/64/128, mathematical
controls, earlier pilots, material datasets, parameter selection or neural fits.
Other settings, data family and all three comparators remain unchanged: eight
repair probes, two outcomes, sixteen proxies, epsilon1e-6, degree cap64, BLAS1.

Timing and memory interpretation
The unmodified run_case includes common posterior/input setup and all kernel,
moment, tail-selection/fallback and score-readout costs as originally recorded.
It does not time a complete end-to-end policy call, action execution, scientific
experiment or general agent. Each method has one ordered timing, not a replicated
performance estimate. Preserve all losing and winning points without filtering.
Keep managed tracemalloc and principal-array workspace labels distinct from RSS.

Largest cached bank:16*2048*2048*8 =536870912 bytes. The conservative temporary
reserve is1610612736 bytes (1.5GiB), not a guarantee. The preparation receipt records
actual Windows memory at source preparation. Immediately before importing the
producer, Windows GlobalMemoryStatusEx must show at least4294967296 available
physical bytes (4GiB); otherwise write PREEXECUTION_RESOURCE_REFUSAL.json and stop
without any case. Do not alter global environment or create a smaller successful
subset automatically. Execution output is exclusive, and completed case receipts
survive any later failure. Root decides a separate recovery action if necessary.

Approval and provenance
The new runner pins both frozen source SHA256s and the prior twelve-case
RESULTS.json SHA cc48fe6581545569760d544da6d5a1342ce36a2ab0855a1fb3fad80787c37893.
It does not change, overwrite or re-read material responses from any prior work.
The single TEMPLATE is non-executable. Root must register actual UTC and exact
runner hash, then approve that registered plan hash with scope
POSTHOC_MOMENT_SCALING_EXTENSION. No producer imports or execution occur during
source preparation or AST-only review. Original mathematical audit receipts stay
bound through the earlier result and are not recomputed here.

Approved command, only after exact root registration and approval:
D:\Anaconda\python.exe run_scaling_extension.py --plan <registered_plan.json>
  --approval <root_approval.json> --output-dir <new_absent_directory>

Preserve successful and failed outputs. No outcome-based source edits, changed
epsilon, selected H/r points, repeated timings, or after-the-fact material claims.
