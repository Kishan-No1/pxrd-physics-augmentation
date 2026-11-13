# A Controlled Analysis of Physics-Informed Augmentation for Powder X-Ray Diffraction Classification

**Kishan Paschapur**, Pranav Kulkarni (Stanford University)
Accepted at IEEE ONCON 2026 · **[Read the paper (PDF)](paper.pdf)**

Models trained on simulated powder X-ray diffraction (PXRD) patterns can do well on synthetic benchmarks and still struggle on measured data. This study keeps a SIMPOD-based Swin-T classifier, its input representation and its data split fixed, and measures what physically motivated augmentation of the training patterns contributes to space-group classification on real measurements.

## Main results

| Model | Synthetic top-5 (24,496 patterns) | Experimental top-5 (20 measured patterns) |
|---|---|---|
| Published Swin (Rincón et al., 2025) | 82.69% | 35%* |
| Clean training, two-fold ensemble | 82.92% | 6/20 (30%) |
| Physics-informed augmentation, two-fold ensemble | 79.63% | **11/20 (55%)** |
| Class-prior baseline (5 most frequent space groups) | — | 7/20 (35%) |

\*The published experimental score is shown for reference only. We could not confirm it was measured on the same set of patterns.

- **Repeated runs:** augmentation improved experimental top-5 in all three fold-0 run pairs, by 5, 2 and 3 patterns.
- **Statistics:** the primary paired comparison gives an exact two-sided McNemar p = 0.0625. The effect is consistent in direction but rests on 20 patterns.
- **Component ablation:** broadening (−3.51 pp) and background (−2.76 pp) contribute most on a shared augmented validation set. Background removal causes the largest drop on the measured patterns (11/20 → 7/20).
- **Reproducibility finding:** a rarest-first fold allocation silently removed the four most common COD space groups from training. The paper documents this pitfall. All reported results use the published split, and no reported checkpoint was selected on the test set.

## Method in brief
- **Simulation:** COD structures simulated with Dans_Diffraction (Cu Kα, 2θ = 5–90°) and rasterized to 521 × 521 radial images, following the released SIMPOD pipeline.
- **Augmentation (training folds only):**
  - Caglioti-type peak broadening, plus size and strain terms
  - Texture-inspired intensity reweighting
  - Polynomial background with an optional amorphous halo
  - Global 2θ shift
  - Poisson counting noise
- **Model and training:** Swin-T (ImageNet-1K weights) with a 230-way head; AdamW. Checkpoints selected on held-out validation data only.

## Limitations
- The 20 measured patterns come from one source.
- Seeds reuse the same observations.
- Each ablation arm is a single run.
- Only one architecture was studied.
- Structure duplicates inherited from the published split limit what synthetic test accuracy says about unseen structures.

## Code and data
- The training and evaluation code will be released with the journal paper (see the pre-registration below). Until then it is available on request (reachtokishan@gmail.com).
- COD is public.
- The 20 measured patterns were provided privately by the SIMPOD authors and are not redistributed here.

<!-- prereg:start -->
## Journal extension: pre-registered external validation

The conference result rests on 20 measured patterns. A journal extension tests it again on a large, independent set of measured patterns from opXRD and RRUFF, scored once under rules fixed and published before any of that data was downloaded.

The extension is by Kishan Paschapur alone (Independent Researcher). Pranav Kulkarni, co-author of the conference paper above, reviewed and approved the pre-registration but is not an author of the extension.

- **Pre-registration:** [`preregistration/PREREGISTRATION_external_validation.md`](preregistration/PREREGISTRATION_external_validation.md), registered on 2026-11-13 at tag `prereg-stage1` and archived on Zenodo: [10.5281/zenodo.24208531](https://doi.org/10.5281/zenodo.24208531).
- **Locked code and inputs:** [`preregistration/SHA256SUMS`](preregistration/SHA256SUMS) records the SHA-256 of every code file and training input the pre-registration locks. The code is released with the journal paper and can be checked against these hashes.
<!-- prereg:end -->

## Citation
```bibtex
@misc{paschapur2026pxrdaug,
  title  = {A Controlled Analysis of Physics-Informed Augmentation for Powder X-Ray Diffraction Classification},
  author = {Paschapur, Kishan and Kulkarni, Pranav},
  year   = {2026},
  note   = {Accepted at IEEE ONCON 2026}
}
```
