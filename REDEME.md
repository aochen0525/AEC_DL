# Acoustic Echo Cancellation with Deep Learning

This project implements deep learning models (including Transformer-based architectures) for Acoustic Echo Cancellation (AEC) on audio signals. It supports training, validation, testing, and real-time inference, with experiment tracking via Weights & Biases (wandb).

---

## Features

- Transformer and Dual-Path Transformer models for AEC
- Mixed precision training for speed and efficiency
- Real-time inference on audio files
- wandb integration for experiment tracking and visualization

---

**Install dependencies**
pip install -r requirements.txt


## Dataset
Use dataset AEC challenge from Microsoft
https://github.com/microsoft/AEC-Challenge.git


## Inference

Enhance a new audio file:

python scripts/inference.py --model_path checkpoints/model_epoch_19.pth --input_path input.wav --output_path enhanced.wav



## License

MIT License



