# Pre-registration: independent external validation of physics-informed augmentation for powder X-ray diffraction space-group classification

**Author:** Kishan Paschapur, Independent Researcher (sole author)
**Status:** Registered at Stage 1 on 2026-11-13 (tag `prereg-stage1`). Any later change is logged in §12.
**Registration:**
- Public commits with tags `prereg-stage1`, `prereg-amendmentA` and `scoring-predictions` in github.com/Kishan-No1/pxrd-physics-augmentation, each archived with a Zenodo DOI.
**Disclosure:** before Stage 1, the author had seen only the published summary statistics of opXRD (Hollarek et al., 2025: 92,552 patterns, 2,179 labelled, contributor list) and of RRUFF (about 1,290 powder patterns). To write the cohort builder before any download, the author also read the opXRD paper's description of each contributor's data and the published source code of its file format (xrdpattern). No pattern or label has been downloaded or viewed.

**Builds on:** K. Paschapur and P. Kulkarni, "A Controlled Analysis of Physics-Informed Augmentation for Powder X-Ray Diffraction Classification," IEEE ONCON 2026 (accepted, ONCON26-000086). Below this is called "the conference paper".

## 1. Purpose

The conference paper found that physics-informed augmentation raised top-5 space-group accuracy on 20 measured patterns from 6/20 to 11/20: 5 augmented-only and 0 clean-only successes, exact two-sided McNemar p = 0.0625, which is the smallest p attainable with five discordant pairs.

Its reviewers noted four problems:
- the measured set was small and was used during development;
- duplicate structures limit what synthetic accuracy shows;
- ablation arms were single runs;
- the contribution needed to be clearer.

This study answers them with a large, independent, multi-source measured cohort scored **once**, in which:
- every model, data rule, hypothesis and analysis is fixed before any new measured pattern is downloaded;
- new models (more seeds, and a second architecture) follow a fixed protocol and are hash-locked before any of them sees measured data;
- results are reported as they come out, whether they confirm, weaken or reverse the conference finding.

## 2. Definitions

| Term | Meaning |
|---|---|
| Pattern | One measured 1D powder diffractogram (2θ, intensity) |
| Cohort | All patterns passing §6.3 after the §6.5 exclusions |
| Structure cluster | A connected component of patterns linked by any of: a pymatgen StructureMatcher match (default tolerances); the same RRUFF mineral name; or the same reduced formula and space group |
| Primary set P | One pattern per structure cluster (§6.5) |
| Twin level | Each pattern's relation to the fold-0 models' training data (§6.6): *train-twin*, *val-twin*, *none* or *ambiguous* |
| Top-k correct | The true space group is among the model's k highest-scoring labels. Ties are broken by the lower class index |
| Seed pair *s* | The clean and augmented fold-0 models trained with seed *s* |
| Draw *N* | The augmentation realization generated with generator seed *N* |
| Arm | One training configuration: clean, full augmentation, or full augmentation with one effect removed |
| Discordant pattern | Top-5 correct for exactly one model in a pair |
| Deviation | Any change to the registered text, hashes, manifest, model list or analysis parameters after their timestamp (§12) |

## 3. Stages

| Stage | Locked | When |
|---|---|---|
| 1 | This document: hypotheses and analysis, data rules, existing model hashes (§5.1), new-model protocol (§5.2), and the hashes of all code as committed (§5.3) | By 2026-11-15, before any external data is downloaded |
| 1b | New models trained per §5.2. No measured pattern of any kind exists in the training environment | 2026-11-16 to 12-04 |
| 2 (Amendment A) | (a) the unlabeled cohort manifest and a separately hashed label file; (b) twin levels and clusters; (c) hashes of every new checkpoint, failed runs included, with run logs; (d) hashes of the cohort builder, wavelength conversion, leakage audit and analysis scripts (first locked at Stage 1, §5.3), plus the analysis outputs on simulated mock cohorts; (e) `uv.lock` and the `.venv` package freeze; (f) the scoring device and precision (fp32, one device for every model); (g) the realized n, stratum sizes, the 30-pattern label-audit result, and the power simulation | By 2026-12-06 (latest 12-10), before any model sees measured data |
| 3 | Blind scoring of the cohort **and the original 20 patterns in one run**. The prediction-file hashes are committed (`scoring-predictions`) before labels are joined; then the analysis runs once | 2026-12-14 to 12-18 |

Amendment A may add only items (a)–(g).

## 4. Hypotheses and error control

**H1 (primary).** On P, Swin-T trained with full augmentation has higher top-5 accuracy than Swin-T trained on clean simulations, pooled over fold-0 seed pairs 0–4.
- *Estimand:* Δ = (1/S) Σ_s [Acc₅(aug_s) − Acc₅(clean_s)], with S = 5.
- *Test:* paired sign-flip randomization test on d_i = (1/S) Σ_s (a_is − c_is), where a_is and c_is are pattern i's top-5 correctness for the augmented and clean models of seed s. The statistic is the paired t of the d_i. p = (1 + #{|T*| ≥ |T_obs|}) / (B + 1), with B = 100,000 and `numpy.random.default_rng(20261115)` (PCG64). Two-sided, α = 0.05. Every resampling procedure (each test and each bootstrap) starts its own generator with this seed, so no result depends on the order in which the procedures run.
- *Interval:* 95% percentile bootstrap over patterns, 10,000 resamples, its own generator with the same seed.
- *Scope:* inference is over patterns, conditional on the five trained pairs. It does not generalize over new training seeds; a per-seed analysis (§9.3) addresses seed variation.
- *Decision:* H1's test passes only if p < 0.05 **and** Δ > 0. Its outcome class follows the rules at the end of this section.
- *Contingency:* if fewer than 5 pairs exist at Amendment A, S is the number available (at least the existing seeds 0–2), recorded there.

**Family 1, fixed sequence:** H1 at α = 0.05; only if H1's test passes, **H2-out** at α = 0.05.
- **H2-out:** Δ > 0 on patterns in P whose twin level is *none*. These measure generalization to structures absent from the training fold. Same estimand and test as H1, restricted to those patterns.
- **H2-in** (twin level *train-twin*) is reported with its interval but is descriptive.

**Family 2, Holm at α = 0.05:**
- **H3a / H3b:** removing broadening (H3a) or background (H3b) lowers measured top-5 accuracy. Δ_r = (1/S) Σ_s [Acc₅(full_s) − Acc₅(r_s)] over ablation seeds 0–4, with the same test and interval as H1. Disclosure: these two effects were chosen as confirmatory **because of** the conference results, so their true effects may be smaller (winner's curse).
- **H5:** a 1D CNN replicates the augmentation benefit. Δ for the 1D models (seeds 0–4) on P, with the same test as H1.

**Error control:** the family-wise error rate is held at 5% within each family, by the fixed sequence in Family 1 and by Holm in Family 2. The two families are not pooled, so across all five confirmatory tests the chance of at least one false positive can exceed 5%. In Family 2, p is the Holm-adjusted p. Intervals are unadjusted 95% percentile bootstrap intervals.

**Not confirmatory:** noise, zero-shift and texture removals; conformal candidate sets; everything in §9.4.

**Smallest effect of interest: 3 percentage points.** In the conference paper, full augmentation lowered synthetic top-5 accuracy by 3.29 points (82.92% to 79.63%, two-fold ensembles). A measured gain smaller than about that would not clearly outweigh the cost.

**Outcome classes.** A confirmatory test *passes* when p < α and the effect is in the predicted direction (Δ > 0). The test alone decides significance; the interval is used only to compare the effect with 3 pp. Each result falls in exactly one class:
- *supported:* the test passes and the interval's upper limit is 3 pp or more;
- *supported but small:* the test passes and the whole interval lies below 3 pp;
- *inconclusive:* the test does not pass and the interval's upper limit is 3 pp or more;
- *contradicted:* the test does not pass and the whole interval lies below 3 pp, which rules out a gain of 3 pp or more.

A significant effect in the wrong direction (p < α, Δ < 0) is classed by the same rules and also reported as a reversal. If the test and the interval disagree about zero (for example, p < α while the interval includes 0), the test decides and the disagreement is reported.

## 5. Models

### 5.1 Existing frozen checkpoints

No existing model is retrained, fine-tuned or recalibrated. The files are in `training/output/checkpoints/`. SHA-256 hashes were computed on 2026-11-03 and re-verified on 2026-11-10.

| Role | Checkpoint | SHA-256 |
|---|---|---|
| Clean, seed 0 | swin_pretrainedTrue_psclean521_fold0_best.pt | e1d05eb3d215c2db0be6763a689d2e3596a257c7eb9c5bac480dac69c52ab83a |
| Augmented, seed 0 | swin_pretrainedTrue_psaug521_fold0_best.pt | a32144e3a21eef1475ce80207dab932a0d77db8aebf4f14a898b6bf09739a1ae |
| Clean, fold 1 (exploratory ensemble) | swin_pretrainedTrue_psclean521_fold1_best.pt | 7376d65419f3a14a8cd2b3dedce4c305277691be6961cd0e7c95804d098c341a |
| Augmented, fold 1 (exploratory ensemble) | swin_pretrainedTrue_psaug521_fold1_best.pt | fbd595d766645da3c48cdf2e5439c7e306f85b55b98ff2bcb8c5a5698451955a |
| Clean, seed 1 | swin_pretrainedTrue_psclean521_fold0_s1_best.pt | 56cf163b0256b6bfb676e63766f8a17fb034ec3b5ffbf5efd4c06a566ef4a5fa |
| Augmented, seed 1 | swin_pretrainedTrue_psaug521_fold0_s1_best.pt | 977a232ae649f6271b0d3720423994dc36f1b8f5caad3ebba48b21ed21c397ae |
| Clean, seed 2 | swin_pretrainedTrue_psclean521_fold0_s2_best.pt | 9edda243d9edeb5a8fc085f72e65287d80a460a3d080c21c8b270827b139e20e |
| Augmented, seed 2 | swin_pretrainedTrue_psaug521_fold0_s2_best.pt | aa2870b8f6d5f248693b7f77a4dc37d32ba65289d802587afed5afb57a0095d5 |
| Clean, alternative selection (exploratory) | swin_pretrainedTrue_psclean521_fold0_bestalt.pt | 1475b301256e41104baa4b6b9fd3beaf688651d4354f8f0daf7e1f668126d667 |
| Ablation, full, seed 0 | swin_pretrainedTrue_abl_full_best.pt | 34be27ef2e6f8b215755f738a9b55bf8b6592c5a1f5a1831157afd48fb7be9fc |
| Ablation, no broadening, seed 0 | swin_pretrainedTrue_abl_no_broadening_best.pt | 1fc3777d429ce5b3fc96d47ee27549fd7ced2a5cf1433f4a9b36836b5c490afa |
| Ablation, no background, seed 0 | swin_pretrainedTrue_abl_no_background_best.pt | efa977e62ed7d01e63d87ebaddbb537dcde95856d5eb633874990f2f0125ffd7 |
| Ablation, no noise, seed 0 | swin_pretrainedTrue_abl_no_noise_best.pt | 2b17ad3a87b915d758ec25f72a02fd465fc44aa3f758231b608a8ca427bd3eeb |
| Ablation, no zero shift, seed 0 | swin_pretrainedTrue_abl_no_zero_shift_best.pt | 146aa51dc9ae50e99b1efb5f02259e2b6fff97985a380fbf4576d709d5969d60 |
| Ablation, no texture, seed 0 | swin_pretrainedTrue_abl_no_texture_best.pt | 15d46546b3257bc456e2d3f1d942cd3e329306dc5fe56f5392b40c6dd43d5784 |

### 5.2 New models: fixed training protocol

**Common to all new runs** (identical to the conference paper):
- AdamW, learning rate 10⁻⁴, weight decay 0.05, label smoothing 0.1, batch size 32.
- 3-epoch linear warmup, then cosine decay over a 30-epoch cap.
- Early stopping: patience 5, minimum improvement 0.15 percentage points, no stop before epoch 12.
- Validation: a 10,000-row fold-1 subsample (`VAL_SUBSAMPLE_SEED` = 20260909). The synthetic test is logged but never selects a checkpoint.
- Every corpus is generated with `--strength 1.0 --compact --clean-from-cache`.
- One GPU type for every new run (planned: RTX 4090, bf16 mixed precision), recorded per run.

**(A) Primary arms, seeds 3 and 4** (4 runs):
- *Clean:* as in the conference paper, with `--seed N`. It also logs the draw-0 augmented validation, for an exploratory alternative-selection checkpoint.
- *Augmented:* draw N (`--seed N`), selecting on the clean fold-1 validation, as for seeds 1–2.

**(B) Ablation arms, seeds 1–4** (24 runs):
- Arms: {full, no broadening, no background, no noise, no zero shift, no texture}, on draw N, using `--disable-effect` for the removal arms.
- The per-effect independent random streams make each removal arm differ from the full arm only in the removed effect.
- All arms select on the **draw-0 augmented** fold-1 validation, the same set as seed 0. Fold 1 holds different structures, so this does not leak.
- `abl_full` seed N and `psaug` seed N train on identical data but select their checkpoints differently. Neither substitutes for the other, and how often their selected epochs agree is reported as a reproducibility check.

**(C) Second architecture: 1D CNN, clean vs augmented, seeds 0–4** (10 runs):
- `train.py --modality 1d`. Clean inputs come from `patterns_f16_papersplit.npy`; augmented inputs from the float16 1D caches of draws 0–4, the same realizations as the images.
- Selection on the clean fold-1 validation (10,000 rows, read from the clean cache). The 1D branch of `train.py` gains this held-out validation before Stage 1; the conference-era 1D path had none.
- Other settings as in `train.py` and `model_1d.py` at their Stage 1 hashes.

**Integrity rules:**
- Every completed run is used. A run is restarted once from scratch with the same seed only for a technical failure: a crash, NaN, or stopping before epoch 12.
- All checkpoints are hashed, failed ones included. No run is dropped, replaced or selected because of any result.
- **Exposure:** before Stage 3, new models may be evaluated only on synthetic data. They are never evaluated on any measured pattern, the conference paper's 20 included.

### 5.3 Code and training-data inputs (SHA-256 at the Stage 1 commit)

Every file below is hashed as it stands at the Stage 1 commit, and `preregistration/SHA256SUMS` in that commit lists the same values. The code is released with the journal paper and can be checked against these hashes then. Compared with the draft table of 2026-11-10: `training/train.py` and `training/dataset_1d.py` changed before Stage 1 for the new training runs of §5.2; 28 files that the draft did not list were added: run-time files imported by the training scripts or run on the cloud pods, and the cohort builder, downloader, Stage 3 scoring and analysis scripts, power simulation and model role map, with their tests.

| File | SHA-256 |
|---|---|
| cloud/run_journal.sh | 24dd4539bf84f43b8d2ff4c1add318b694b5cf8c03b9dc04a42a6cb91ea57a19 |
| cloud/setup_instance.sh | 0f0377aae9e424e0092de88b7ccc45a722b00674ea70772b47b3865b863908a7 |
| cloud/sync_journal.sh | 029e92d8c024fac68f4be4b54ef5909035a1d1ed7b12c05916592c10b25d4908 |
| cohort/build_cohort.py | c2abe258f2181e72b6b763955c9e618319bf0bced32976d3779901dfee0b888e |
| cohort/cu_convert.py | 935ef7983f75f350cb5536ccdb4ae44d278ead8cb2d3542ec00f6c531354a654 |
| cohort/download_measured.py | f40698f074c035eaeb2718892600ce19f42e3b71b82180cf37213e4e2e7d2096 |
| cohort/labels.py | 1a3a5a71ed6aea7c83bc2e765bded926a87e55a89d9201b41b93f63fff2645f4 |
| cohort/pyproject.toml | e19470b4f94b4e9a37fc3a6e26ca627688cf8729bd629f9a7925cecd8de5e210 |
| cohort/sources.py | 95ecc324bfbe87a58f1aa74c68915de0bccbc4b5ebcc04845fe04741ecafcc68 |
| cohort/tests/conftest.py | 7ddbdb90ca59218a834b39c3d2505916b27df112ed8c043985b0bcef49bdb22c |
| cohort/tests/synth.py | 3d2806c1384b769aee58a9a36fa9499daa4c44a8d75464f021c56f1e3774c20f |
| cohort/tests/test_build.py | 7f6070c151a300b9c67f66c9b6e85b041547a9bf9520eabde051743ad4314f16 |
| cohort/tests/test_cu_convert.py | 3763a6b44acf2021f2a68b30c204fcf67a21bd8e706997ed5b6e1570806d2b7b |
| cohort/tests/test_labels.py | 58a22eed9fbdf8c2389ea27ae7db23fea10397f97f674d1af89a7fbbea6617e4 |
| cohort/tests/test_twin_levels.py | f0167130626080b8682d3477cdebcc6c1323b039dc05f02e39441441d1dc5aa0 |
| cohort/twin_levels.py | ec45af82f40d368bc08dacb1c77ea2a236d37eb1364f323249825b75eaf3fd25 |
| cohort/uv.lock | c47ed3e054add2160d83b8ec617be53737a97a28dae110c381b5d06bf325440a |
| preprocessing/scripts/generate_augmented_radial_images.py | 5e3e9e7c258c473c31092ecfe2ca6d23ca0f1df2c0ee24b05626dbe4efef5bf3 |
| preprocessing/simpod/__init__.py | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| preprocessing/simpod/augment.py | 390410d9ee72326f19c6d2735ee557670f56a5960f2098d477af5fbca1817dba |
| preprocessing/simpod/augment_config.py | 2b4e0083c7eeee3abd6f1e65113f797506d3bffd2667d791f6d20008604c33bd |
| preprocessing/simpod/config.py | 13fef876cc9891f754fbc86cca91275ed80a40be955e3d24f11d01e6a22c9509 |
| preprocessing/simpod/radial_image.py | 0a1f7aa16894f02fb13f28e90af0eb3033206976556bab71ac765217fd3e5e12 |
| preprocessing/simpod/xrd_simulate.py | 2ec38248bf0330833ceb7adfd74b7afce045600dfcbd0390e74fb61f00a9ba39 |
| training/build_augmented_folds.py | adf26dc6644add96ad03a58c8a1220fa9273d6db418040f3560be7b3dd937555 |
| training/build_real_dataset.py | cf27077843252e11cdfe3a666c7c92945fdcd230ee6e5dbd5d8a2301a832660a |
| training/cache_patterns.py | e4f9ac7773e15d11c07193bee644d2eddfdcccf6fd37a26c034fc85cb428571f |
| training/config.py | a1e02d8eec7d993fe5e7a219a36f19f4d45bc36e728b93211e04059c20023a65 |
| training/dataset.py | 3995846ec1b58255b39766a53828b46c88ad7f55350bc7bbcf6e0bb486154da4 |
| training/dataset_1d.py | 4dca53408a704143be60a2cf72dbc68bb60fd82fe174f909f4cabf0467b936f4 |
| training/eval_utils.py | 2adc0d74bde3b9b5be9cf7217956a15b7ad7ece011af4da089b8c2b3600ec544 |
| training/evaluate_ensemble.py | 0bfe9145f8c302371e4ace7cdd0faaf2c18cb85ee66307cc5f83d44da570d487 |
| training/evaluate_real.py | 494219777ac64657ed130c17c943d2383f803be0f0bfef2d54292612ea3dc017 |
| training/model.py | 5882aa68e2a3ad4c4a19b4589ad8b2ee17503fa5df74b941314bda2514cf493d |
| training/model_1d.py | d49501643423e41bd9a276a31d93a0b0ad0a3d312ae3b8d8b3e5f90104a6c3d5 |
| training/paired_test.py | b9449c4ebc0b3c6fc7d8cea214148734140e73ae1ee3ec0de485d5614f4de1e1 |
| training/power_sim.py | 42959ebd7d1e8030426c508f16300b9a4992b2e5d061cbc9bfa928a3b0cc2684 |
| training/prereg_models.csv | 9784ce5ca39f907edc6c0306ba366d14f2bdbe20bf7a5f52b3e83784900d4730 |
| training/prereg_stats.py | 75afcced966ccb8db61cd10e13ffe27fb51ef5b19ee8246154bbc2c51361d206 |
| training/real_baselines.py | 266c52d4cab1cd1dc8e2d556357d7dd0088d983cca4074af8ff918ac576b78a3 |
| training/reroot_paths.py | 432ccc9cbad4f35bad85e59bcd9cd99052f0f1398a22378949f9553f429c7713 |
| training/run_analysis.py | 6e35233c5e802fcb6ddbe1024cfa1ec8f4ce53afa4f708aefa2623e80637b8ee |
| training/score_cohort.py | db0b15703e11810aa2a056e63e901194649ef025a0f73ebf6b38c927f8cfd2ae |
| training/split.py | 01684971dfc07cae27d0fc57973b755ba722dbdd407ac7815edc5579832dada2 |
| training/tests/test_prereg_stats.py | 0fabf667851d8eef356325d18d0d6fcca3c0761142e7e43a2271f8902a66098c |
| training/tests/test_stage3.py | 1318faa6a79f629079618e9b95f9e55170d51ee358f5e405665ce916dfffc403 |
| training/train.py | 9eca8b1b147059a8273238b0a91b17e712a73b1d531636419c4eac444ea13bff |

Training-data inputs of the new models (hashed, not published):

| File | SHA-256 |
|---|---|
| training/output/folds_papersplit.parquet | f24a583e5daf0c91237c145a3c1263f0aaed9c3e656499030cbd93efed0d4aa3 |
| training/output/patterns_f16_papersplit.npy | d8ff59182e1bd94954acfa9d33eddd8d142ab6718145b6dbe543c0c72d08ea98 |
| training/output/patterns_f16_papersplit.order.parquet | 05c0258f962c47f283b778921b774a2681a8970aa6a10f15c8a51f2a9852723a |

### 5.4 Model input for measured patterns

- Image models receive the float radial image built by `pattern_to_model_input` in `evaluate_real.py`, exactly as in the conference paper's measured-pattern evaluation. It is **not** quantized to 8 bits, unlike the training PNGs.
- A sensitivity analysis quantizes as in training, `np.round(img * 255) / 255` (`radial_image.py`).
- 1D models receive `pattern_to_model_input_1d`.
- Scoring uses fp32 on one device for every model.

## 6. Data

### 6.1 Sources
- **opXRD** (Hollarek et al., Zenodo record 15298026). The version, file name and MD5 are recorded at download. A mismatch with the version noted when this plan was drafted (v11, MD5 e8e9800a8ecd2b7f01a764fdf77d5c53) is logged, not fixed.
- **RRUFF** measured powder files (rruff.info). `Powder__Xray_Data_XY_RAW` is preferred. `XY_Processed` is used only when no RAW file exists; it is flagged and reported as its own group, because processing may include background correction.

### 6.2 Access protocol
- Data is downloaded after the Stage 1 commit. Time, URL and hash are logged.
- Cohort decisions use only metadata and labels. No model, model output or visual judgement of pattern quality is involved. Parse failures are logged automatically.

### 6.3 Inclusion rules
A pattern is included if all of these hold:
1. **Measured X-ray powder diffraction.** Excluded:
   - calculated patterns (RRUFF `DIF_File`; headers saying XPOW, ATOMS or "calculated");
   - neutron or electron data;
   - entries flagged as thin-film or grazing-incidence;
   - **whole contributors** whose documentation describes thin-film, grazing-incidence or in-situ series (for opXRD, LBNL and EMPA are excluded on this basis).
2. **Labelled single-phase.** A RRUFF specimen counts as single-phase; an opXRD entry must list exactly one phase.
3. **A space-group label** that maps to an International Tables number 1–230 through the conference paper's Dans_Diffraction Hermann–Mauguin conversion. Symbols in a non-standard setting (such as P2₁/n or Pbnm), which that conversion's 230 standard symbols do not include, map through gemmi's space-group table. If only a CIF is provided, the number comes from the CIF via gemmi. Missing, conflicting or unparseable labels are excluded.
4. **A known wavelength:** from the entry's metadata, or from the opXRD paper's documented source for that contributor's instrument. RRUFF patterns are excluded when neither the XY file nor the same sample's DIF file states a wavelength. An exploratory sensitivity analysis includes them, assuming Cu Kα.
5. **Enough data after conversion to the Cu Kα₁ grid (1.5406 Å):** at least 50% of 5–90° covered and at least 200 measured points inside it.

### 6.4 Labels
- RRUFF space groups come from the same sample's RRUFF record: the SPACE GROUP line of its DIF file. A sample without one has no label (§6.3 rule 3).
- The label source is recorded per pattern.
- For opXRD patterns linked to a COD ID, we record whether that COD entry's space group equals the cohort label.
- **Label audit:** before scoring, 30 randomly chosen included patterns (`default_rng(20261115)`) are checked against their source records. The error rate goes into Amendment A.

### 6.5 Exclusions and de-duplication
- **Independence:** patterns whose reduced formula and space group match one of the conference paper's 20 patterns are excluded from confirmatory analyses. The original 20 are scored in the same Stage 3 run and reported separately as the historical development set.
- **The same measurement in both sources:** RRUFF's copy is kept.
- **Repeated or in-situ series** (if any survive rule 1): among one contributor's patterns of the same compound and label that state a temperature or are flagged in situ, the first measured at ambient conditions (283–308 K, or no temperature stated). A pattern of that compound with neither is a separate measurement.
- **Primary set P:** one pattern per structure cluster: the widest 2θ coverage; ties go to RRUFF before opXRD, then to the lowest source ID. A sensitivity analysis uses all included patterns.

### 6.6 Twin levels (against the fold-0 models' training data)

| Level | Rule |
|---|---|
| *train-twin* | The pattern's COD ID is a fold-0 row, or its structure matches a fold-0 structure with the same reduced formula under StructureMatcher |
| *val-twin* | The same tests match only fold-1 structures |
| *ambiguous* | Only a formula-plus-space-group match exists (no structure available), or a solid-solution or partially occupied composition prevents a match. **All opXRD CNRS patterns (extracted from COD) are at least ambiguous and never *none*** |
| *none* | No match at any level |

Levels are frozen in Amendment A. H2-out uses *none*; H2-in uses *train-twin*.

### 6.7 Size and power
Amendment A reports:
- n(P) and the size of each twin level;
- a power simulation for H1 and H2-out at the realized n, using the conference paper's discordance rate (25%) and no outcomes, with seed outcomes independent and fully shared as two bounds (`training/power_sim.py`).

If n(P) < 100, or n(*none*) < 30, the affected test is still run and is labelled underpowered.

## 7. Preprocessing

1. **The cohort builder writes Cu-equivalent files:** two-column whitespace text, 2θ in degrees, increasing.
   - Non-Cu data is converted through d-spacing: d = λ_src / (2 sin θ_src), then 2θ_Cu = 2 arcsin(1.5406 / (2d)). Points with 1.5406 / (2d) > 1 are dropped.
   - Cu Kα data is not converted, and Kα2 is not stripped.
   - **No Lorentz-polarization, absorption or intensity correction is applied.** Converted patterns therefore carry distorted relative intensities and 2θ peak widths. That is reported, and LP reweighting is an exploratory check.
   - The conversion is unit-tested on simulated patterns and hashed at Stage 1 (§5.3).
2. Everything else is identical to `evaluate_real.py` at its hash, which stays unchanged:
   - interpolation onto the 10,824-point 5–90° grid;
   - boundary-value fill (zero fill as a sensitivity check);
   - max normalization;
   - nearest-neighbour downsampling to 1,250 points;
   - 521 × 521 radial rasterization (float);
   - no background subtraction in the primary analysis.

## 8. Outcomes

Per pattern and per model: top-1 and top-5 correctness (ties broken by the lower class index) and the full softmax vector, stored for the exploratory analyses.

## 9. Analysis plan

**9.1 Confirmatory:** H1, H2-out, H3a, H3b and H5, as in §4. Which checkpoint fills each arm and seed is fixed in `training/prereg_models.csv`, and the analysis is `training/run_analysis.py` (both hashed in §5.3).

**9.2 Sensitivity analyses of H1** (reported regardless):
- a two-way bootstrap that resamples both patterns and seed pairs;
- the per-seed Δ_s and the number of seeds with Δ_s > 0;
- a sign-flip test clustered by contributor (labs are not independent). RRUFF counts as one contributor, and with k contributors the smallest attainable p is 2/2ᵏ, so this test is reported with that floor;
- all included patterns instead of P;
- Cu-native patterns only: the patterns in P measured with Cu Kα radiation, which §7 leaves unconverted, so the result does not rest on the wavelength conversion;
- zero fill;
- 8-bit input.

**9.3 Reported regardless of outcome:**
- top-1 and top-5 accuracy with counts and Wilson 95% intervals for every model and seed;
- the class-prior baseline (always predicting 14, 2, 15, 19, 4) and a cohort-frequency reference;
- the **continuity analysis**: the seed-0 pair, exact two-sided McNemar test, paired difference with a 95% Newcombe hybrid score interval (Newcombe 1998, method 10; no continuity correction on φ);
- H2-in;
- per-source results (opXRD by contributor, RRUFF RAW vs Processed);
- the cohort description (sources, contributors, radiation, coverage, space-group and crystal-system distributions, label sources, label-audit error rate);
- synthetic test accuracy for every model on the full test set and on the deduplicated subset (excluding the 312 rows with a structure_key or pattern_key collision);
- the synthetic cost of augmentation as a seed-level mean with a t interval (a trade-off, not a test);
- ablation validation deltas as mean ± SD over seeds 0–4.

**9.4 Exploratory** (labelled, no confirmatory claims):
- **Conformal candidate sets:** split-conformal LAC (score = 1 − softmax of the true class); calibration of 30%, stratified by source; finite-sample quantile ⌈(n_cal + 1)(1 − α)⌉/n_cal; α = 0.10; repeated over 1,000 random splits. We report the median and the 2.5–97.5% range of coverage and mean set size, plus the empty-set rate, for each model. APS sets as well.
- Calibration: expected calibration error and reliability diagrams.
- Noise, zero-shift and texture removals.
- Breakdowns by radiation (Cu vs converted), coverage bin, crystal system and label source.
- SNIP background subtraction; LP reweighting.
- Fold-0 plus fold-1 ensembles; the alternative-selection clean models.
- RRUFF patterns without a stated wavelength, included as Cu.
- The original 20 pooled with P.
- A comparison of augmentation ranges with statistics measured on the cohort (peak FWHM, background fraction, signal-to-noise), computed after scoring and never used for tuning.

**9.5 Interpretation rules:**
- Effect sizes and intervals lead, and are classified against the 3 pp smallest effect of interest.
- "Significant" is used only for the confirmatory tests at their α.
- A non-replication or reversal is reported as such.
- Stratum results are not generalized.
- Classification accuracy is never called measurement accuracy.

**9.6 Missing data:** included patterns that fail to parse are listed and excluded before any model output exists.

## 10. Single-pass and blinding protocol

1. Scoring (`training/score_cohort.py`) reads only the unlabeled manifest (cohort plus original 20) and writes per-model prediction files.
2. Their SHA-256 hashes are committed and tagged (`scoring-predictions`) before the label file is opened.
3. Labels are joined and the hash-locked analysis scripts run once.
4. No pattern is added or removed after predictions exist. No preprocessing or analysis setting is changed.
5. If a bug is found after scoring, both the original and the corrected results are reported, and the fix is logged in §12.

## 11. Data and code availability

The cohort manifest, label sources, per-pattern predictions, all new code and new checkpoints, and this document are released with the journal submission (GitHub + Zenodo). opXRD material is redistributed only as CC BY 4.0 allows. RRUFF files are referenced by ID. The conference paper's 20 private patterns are not redistributed.

## 12. Deviations log

| Date | Change | Reason | Effect on results |
|---|---|---|---|
| | | | |

## 13. Contributions and AI assistance

- Kishan Paschapur (sole author) conceived and designed this study and runs the experiments and the analysis.
- Pranav Kulkarni (Stanford University) mentored the conference paper on which this study builds, through the Echelon Scholars program, and reviewed this design and approved its registration on 2026-11-13. Pranav Kulkarni is not an author of this pre-registration or of the journal paper.
- Claude (Anthropic) assisted with code, analysis design and drafting. The author is responsible for all content.

## 14. Cut order if time runs short (decided before any data)

The first item is cut first. Every cut is logged as a deviation that doesn't depend on data, made before scoring.

1. Exploratory extras: calibration, APS sets, ensembles, crystal-system breakdowns.
2. Conformal sets.
3. H5 and the 1D runs: **withdrawn, not tested,** if they aren't hashed at Amendment A.
4. Ablation seeds 3–4 (H3 then uses seeds 0–2).
5. A source not ready by 2026-12-02: deferred to a later replication cohort.
6. Primary seeds 3–4 (H1 then uses seeds 0–2).

**Never cut:** H1, the blind prediction-hash commit, single-pass scoring.
