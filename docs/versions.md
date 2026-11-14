# Locked versions (do not change without re-exporting everything)

- ExecuTorch: 1.5.1 (Python package and org.pytorch:executorch-android:1.5.1 in the app)
- torch: 2.11.0, torchvision: 0.26.0
- Export notebook: training/export_vision.ipynb (Kaggle, Linux, Python 3.13)
- vision.pte: MobileNetV3-Small, 13 classes (classes_v3.json), input 1x3x224x224, XNNPACK, 6.16 MB
- Python check of the .pte failed with "tensor does not have a device" (a Python binding problem); the file is verified on the phone instead
- Use the SAME ExecuTorch version for the Llama export and the Android app
