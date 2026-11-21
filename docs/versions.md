# Model version history

Only v3 is deployed (`export/vision.pte`). Earlier notebooks are not kept in this repository.

| Version | What changed | Result | Decision |
|---|---|---|---|
| v1 | Random split, lab photos only | 99.02% on a lab test | Misleading: near-identical photos appeared in both train and test |
| v2 | Split that keeps near-copy photos together, plus an "Other" class | Lab 99.1%, real field photos 23.5% | Dropped: lab accuracy did not carry over to real photos |
| v3 | 13 classes, group-aware split, PlantDoc field photos added, about 70/30 lab/field sampling per batch, strong augmentation | Lab val 98.1%, field val 59.2%, lab test 97.6%, untouched field exam 44.6% (45/101) | **Deployed** |
| v4 | MobileNetV3-Large, label smoothing 0.1 | Lab val 99.1%, field val 61.5%, lab test 99.1% | Not adopted: no clear gain on real photos for a larger model |

## Confidence threshold (v3, on the 179 field validation photos)

These numbers are optimistic because the threshold was chosen on the same photos.

| Threshold | Share of photos answered | Accuracy on those |
|---|---|---|
| 0.5 | 89% | 61.9% |
| 0.6 (used) | 77% | 67.9% |
| 0.7 | 69% | 71.0% |
| 0.8 | 60% | 71.3% |
| 0.9 | 51% | 74.7% |

Class order is in `training/classes_v3.json`. The 101 exam photos were never used for training or tuning.