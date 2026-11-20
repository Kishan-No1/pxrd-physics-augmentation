# Amendment A (pre-registration §3, Stage 2)

Registered on 2026-11-20 (tag `prereg-amendmentA`), before any model saw a measured pattern. Hashes only (D-01): the
cohort files, labels and checkpoints stay with the author and are released with the paper. Every hash is in
the files listed in `SHA256SUMS`.

**(a) Cohort manifest and labels.** Final build 2026-11-20 (provisional: False).
Cohort 1759 patterns; unlabeled manifest SHA-256 `51c88eee2da7a812cca03e2e2593d0439773be5e5442bfaa74dc4f8c19673adc`; label file `labels.csv`
SHA-256 `c7ba0707f63bea87ca5c02d87f4a237ca0088ee11475fef4c08332fabc5b50f0`, hashed separately and not opened until the prediction hashes are tagged (§10).
Exploratory set (§12): 0 patterns (not scored).

**(b) Twin levels and clusters.** P = 1104 patterns in 1104 clusters; twin levels in P:
ambiguous 700, none 327, train-twin 39, val-twin 38. `twin_levels.csv` SHA-256 `745ac79d1fe8c8d5645eee90a2ae83efe8e0136f9b5718632729f436b14f8270`.

**(c) New checkpoints.** 38 finished runs (`checkpoints_new.tsv`, every file with its SHA-256 and the pod it
trained on); other attempts: none. Run logs: `run_logs.sha256`. The 15 frozen §5.1
checkpoints re-verified: 15 of 15 match.

**(d) Code.** `code.sha256` hashes every code file; `code_status.tsv` marks each one:
new after Stage 1 57, registered, changed since Stage 1 (§12) 6, registered, unchanged 44. Changes to registered files are in §12. Analysis outputs on
a simulated mock cohort with planted effects (0, +5, −5, +30 pp, and the seeds 0–2 / no-1D contingency):
`mock_outputs_SHA256SUMS`.

**(e) Environments.** `cohort_uv.lock` (cohort builder); `freeze_scoring.txt` and `freeze_cohort.txt`
(`uv pip freeze` of both environments). The scoring environment has no lock file; its freeze is the record.

**(f) Scoring device.** mps, fp32, one device for every model; torch 2.14.0,
Python 3.13.9, Apple M5 Pro. Rehearsal on simulated patterns: repeat run bit-identical
True; CPU gives the same top-5 for every model and pattern
True (max softmax difference 1.722574234008789e-05).

**(g) Size, strata, audit, power.** n(P) 1104 (by source {'rruff': 639, 'opxrd': 465}; Cu-native 893);
n(none) 327. Label audit: 0 errors in 30.
Power (power_sim_n1104_none327.json):
  - independent Δ = 3.0 pp: H1 0.992, H2-out 0.667
  - independent Δ = 5.0 pp: H1 1.0, H2-out 0.978
  - independent Δ = 8.0 pp: H1 1.0, H2-out 1.0
  - shared Δ = 3.0 pp: H1 0.469, H2-out 0.107
  - shared Δ = 5.0 pp: H1 0.895, H2-out 0.393
  - shared Δ = 8.0 pp: H1 0.999, H2-out 0.81
