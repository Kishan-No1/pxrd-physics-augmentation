# A Controlled Analysis of Physics-Informed Augmentation for Powder X-Ray Diffraction Classification

**Kishan Paschapur**, Pranav Kulkarni (Stanford University)
Under review at IEEE ONCON 2026 · **[Read the paper (PDF)](paper.pdf)**

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
- The training and evaluation code is available on request (reachtokishan@gmail.com) and will be released after review.
- COD is public.
- The 20 measured patterns were provided privately by the SIMPOD authors and are not redistributed here.

## Citation
```bibtex
@misc{paschapur2026pxrdaug,
  title  = {A Controlled Analysis of Physics-Informed Augmentation for Powder X-Ray Diffraction Classification},
  author = {Paschapur, Kishan and Kulkarni, Pranav},
  year   = {2026},
  note   = {Under review at IEEE ONCON 2026}
}
```
