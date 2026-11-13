# Locked versions (do not change without re-exporting everything)

- ExecuTorch: 1.6.0
- torch: 2.15.0
- torchvision: 0.29.2
- Export notebook: training/export_vision.ipynb (Kaggle, Linux, Python 3.13)
- vision.pte: MobileNetV3-Small, 13 classes (classes_v3.json), input 1x3x224x224, XNNPACK, 6.16 MB
- Check: max abs difference vs PyTorch = 1.26e-05, same top class
- Use the SAME ExecuTorch version for the Llama export and the Android app.
