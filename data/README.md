# Data

The datasets are not included in this repository (size and licences). To retrain, add them as Kaggle datasets and attach them to the training notebook.

- **PlantVillage** (lab photos of single leaves on plain backgrounds): tomato and potato classes only.
- **PlantDoc** (real field photos from the web, noisy labels): used for the 30% field share of each batch, the 179-photo validation set and the untouched 101-photo exam.

`training/splits_v3.json` lists which photo went to train, validation and test, so the experiment can be repeated.