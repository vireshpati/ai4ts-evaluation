# AI4TS Pipeline

## Evaluation and Inference

After training a model, you can evaluate it on the test set or run inference on individual audio files:

### Evaluating on the Test Set

To evaluate your trained model on the test set:

```bash
python evaluate.py
```

Options:
- `--num-classes`: Number of classes (10 or 35), default: 10
- `--batch-size`: Batch size for evaluation, default: 32
- `--data-dir`: Custom data directory (if not using the default)
- `--checkpoint`: Path to a specific checkpoint (default: uses best checkpoint)

Example:
```bash
python evaluate.py --num-classes 10
```

### Running Inference on a Single Audio File

To classify a single audio file using your trained model:

```bash
python infer.py --audio-file path/to/your/audio.wav
```

Options:
- `--audio-file`: Path to audio file (.wav) to classify (required)
- `--num-classes`: Number of classes (10 or 35), default: 10
- `--checkpoint`: Path to a specific checkpoint (default: uses best checkpoint)
- `--data-dir`: Custom data directory (if not using the default)

Example:
```bash
python infer.py --audio-file data/speech_commands/yes/00f0204f_nohash_0.wav
```

This will display the predicted class and confidence scores. 