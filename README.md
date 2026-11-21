# Offline Crop Disease Detector (Tomato and Potato)

An Android app that identifies tomato and potato leaf diseases **fully offline** on a mid-range phone (Moto G34 5G) and gives farmers simple, verified advice in English and Tamil. Built for the **Bharat AI-SoC Challenge, Problem Statement 1** (offline crop disease detection).

A farmer photographs one leaf. A small vision model on the phone names the disease. The app shows verified advice from a curated knowledge base, and an on-device **Llama 3.2 1B** model rewrites that advice in simple words. No internet, no cloud, no data leaves the phone.

## Features

- **Offline from start to finish.** Vision model, language model and knowledge base all run on the phone. It works in airplane mode.
- **13 classes:** 12 diseases and healthy leaves across tomato and potato (list below).
- **Honest "Not sure".** If the vision model's confidence is below 0.6, the app says so and asks for a better photo instead of guessing.
- **Verified advice first.** Symptoms, cause, treatment and prevention come from a reviewed knowledge base (`knowledge_base/diseases.json`), in English and Tamil, with an English / Tamil toggle.
- **Llama summary with a safety checker.** Llama 3.2 1B (via ExecuTorch) summarises only the verified facts in about 80 words. An output checker rejects replies that contain doses or numbers not in the facts, unknown words, repeated text, a wrong disease name, or a missing "ask your local agriculture officer" line. A rejected reply is replaced by a safe message, and the verified advice stays on screen.
- **Input check.** A simple colour check rejects photos that contain almost no plant colour.

### Classes

Potato: Early blight, Late blight, Healthy.
Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Mosaic virus, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Healthy.

## How it works

```
Photo (gallery or camera)
   |
   v
Colour check (enough plant colour?) --no--> "This does not look like a leaf"
   |
   v
MobileNetV3-Small, ExecuTorch .pte (XNNPACK), 1x3x224x224
   |
   +-- confidence < 0.6 --> "Not sure" + top-3 guesses
   |
   v
Detected disease
   |
   +--> Verified knowledge-base advice (English / Tamil), always shown
   |
   +--> Llama 3.2 1B (ExecuTorch, INT4) rewrites the verified facts in simple words
            |
            v
        Output checker: pass -> show; fail -> safe message
```

| Part | Choice |
|---|---|
| Vision model | MobileNetV3-Small, fine-tuned, exported with ExecuTorch 1.5.1 (XNNPACK), 6.2 MB |
| Language model | Llama 3.2 1B Instruct, SpinQuant INT4 (pre-exported `.pte`), run with ExecuTorch |
| App | Kotlin and Jetpack Compose, adapted from Meta's ExecuTorch Llama demo app |
| Knowledge base | `knowledge_base/diseases.json`: 13 entries, English and Tamil |

## Results

Measured on held-out photos. Lab photos come from PlantVillage; field photos come from PlantDoc (real-world, noisy labels).

| Test | Result |
|---|---|
| Lab test set (2,232 photos) | 97.6% |
| Lab validation (2,233 photos) | 98.1% |
| Field validation (179 photos) | 59.2% |
| **Untouched field exam (101 photos, run once)** | **44.6% (45/101)** |

Per-class result on the 101-photo field exam:

| Class | Correct |
|---|---|
| Tomato Yellow Leaf Curl Virus | 13/15 |
| Tomato Late blight | 6/10 |
| Tomato Septoria leaf spot | 6/12 |
| Tomato Bacterial spot | 4/9 |
| Tomato Early blight | 4/9 |
| Potato Early blight | 3/14 |
| Tomato healthy | 3/8 |
| Potato Late blight | 2/8 |
| Tomato Leaf Mold | 2/6 |
| Tomato Mosaic virus | 2/10 |
| Potato healthy, Target Spot, Spider mites | no exam photos |

Lab accuracy is high, but accuracy on real field photos is much lower. The 101-photo exam has about +/-10 points of noise. Details and version history are in `docs/versions.md` and `benchmarks/`.

Rough on-device numbers seen during testing (not a formal benchmark): vision about 45 to 235 ms per scan; Llama 1B about 10 to 13 tokens per second, so a summary takes roughly 7 to 18 seconds.

## Limitations

- **Real-field accuracy is 44.6%** on the untouched exam. Treat a result as a first hint, not a diagnosis.
- **Tomato Leaf Mold** is often "Not sure" (2/6 in the exam).
- **Green non-leaf scenes** (for example grass or turf) can pass the colour check and receive a wrong disease. A "Not a leaf" class in the vision model would fix this and is future work.
- **Llama is loaded by hand** each time the app starts (Select a Model, then Load Model) before scanning. Automatic loading crashed during development and was dropped.
- Llama's wording can be slightly stronger than the verified text (for example "use insecticides"). The verified advice above it is the reference, and the app adds an "ask your local agriculture officer" line.
- The Tamil text has not yet been checked by a native-speaking reader.
- No severity level, scan history or region-specific advice yet.
- Exam photos exist for only 10 of the 13 classes.

## Repository layout

```
app/              Android app (Kotlin, Jetpack Compose; open this folder in Android Studio)
export/           vision.pte, the final vision model (ExecuTorch 1.5.1, XNNPACK)
knowledge_base/   diseases.json, verified advice (English and Tamil) with sources
training/         Kaggle notebooks: train_vision_v3.ipynb, export_vision.ipynb;
                  classes_v3.json (class order), splits_v3.json (train/val/test split)
benchmarks/       measured results and confusion matrix
data/             how to get the datasets (not included)
docs/             version history
NOTICE            credits and third-party licences
```

## Setup and run

**Requirements:** Android Studio, a phone with USB debugging (tested on Moto G34 5G, Android, arm64), `adb`.

1. **Get the Llama model** (not included because of its licence and size). Accept Meta's Llama 3.2 licence, then download the pre-exported `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` and `tokenizer.model` from the official ExecuTorch / Meta Llama 3.2 1B SpinQuant release.
2. **Push the models to the phone:**
   ```
   adb push export/vision.pte /data/local/tmp/vision.pte
   adb shell mkdir -p /data/local/tmp/llama
   adb push Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte /data/local/tmp/llama/
   adb push tokenizer.model /data/local/tmp/llama/
   ```
3. **Build and install:** open `app/` in Android Studio, run Build > Assemble Project, then run the app on the phone.
4. **Load Llama, then scan.** In the app choose Select a Model: backend XNNPACK, the Llama `.pte`, `tokenizer.model`, Data Path empty, Model Type LLAMA_3, then Load Model. After that, send a leaf photo.
5. **Offline check:** turn on airplane mode and scan again. The results are the same.

## Training and export

- Notebooks run on Kaggle (Python 3.13). Pinned versions: ExecuTorch 1.5.1, torch 2.11.0, torchvision 0.26.0.
- `training/train_vision_v3.ipynb` fine-tunes MobileNetV3-Small on PlantVillage plus PlantDoc (about 70% lab and 30% field photos per batch, strong augmentation, group-aware splits listed in `splits_v3.json`).
- `training/export_vision.ipynb` exports the trained weights to `vision.pte`.
- Datasets: see `data/README.md`.

## Credits

- Meta **ExecuTorch** and its Llama demo app (BSD-licensed), which the app is adapted from.
- **Llama 3.2 1B Instruct** by Meta, under the Llama 3.2 Community License.
- **PlantVillage** and **PlantDoc** datasets.
- Disease advice compiled from public agricultural extension sources, listed per entry in `knowledge_base/diseases.json`.

See `NOTICE` for details.

Author: Viroshan (GitHub: Viroshan2307).