# AI4TS Benchmarking Pipeline

A modular benchmarking pipeline for testing different model architectures on time series and sequence tasks, leveraging the fairseq framework.

## Setup

1. Install requirements:
```bash
pip install fairseq torchaudio numpy pandas matplotlib tqdm
```

2. Download the Speech Commands dataset:
```bash
mkdir -p data/speech_commands
cd data/speech_commands
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
cd ../..
```

## Training a Model

Use the `train.py` script to train a model:

```bash
# Train on SC10 (10-class Speech Commands)
python train.py --num-classes 10 --max-epochs 20 --batch-size 32 --data-dir data/speech_commands

# Train with custom hyperparameters
python train.py --num-classes 10 --max-epochs 30 --batch-size 64 --lr 0.0005 --data-dir data/speech_commands

# Train in debug mode (faster, with a smaller dataset)
python train.py --num-classes 10 --max-epochs 5 --batch-size 32 --debug
```

Models are saved to `checkpoints_speech_<num_classes>class/`.

## Evaluating a Model

Use the `benchmark.py` script to evaluate a trained model:

```bash
# Evaluate a trained model on the test set
python benchmark.py --data-dir data/speech_commands --checkpoint checkpoints_speech_10class/checkpoint_best.pt --split test

# Evaluate on validation set
python benchmark.py --data-dir data/speech_commands --checkpoint checkpoints_speech_10class/checkpoint_best.pt --split valid

# Evaluate with different model architecture settings
python benchmark.py --data-dir data/speech_commands --arch simple_linear_transformer_arch --encoder-layers 6 --encoder-embed-dim 512 --encoder-attention-heads 8
```

Results are saved to the `benchmark_results/` directory.

## Comparing Multiple Models

After benchmarking multiple models, use `benchmark_comparison.py` to analyze and compare the results:

```bash
# Compare all benchmark results
python benchmark_comparison.py

# Compare only results for a specific task
python benchmark_comparison.py --task speech_commands

# Compare only test split results
python benchmark_comparison.py --task speech_commands --split test

# Save results in different formats
python benchmark_comparison.py --format csv,html,plot --output reports/sc10_comparison
```

## Modular Pipeline Architecture

This pipeline is designed to be modular and extensible:

1. **Models**: Create new model architectures by adding them to the `models/` directory.
2. **Tasks**: Add new tasks to the `tasks/` directory.
3. **Benchmarking**: The benchmarking system works with any registered model and task.

### Adding a New Model

1. Create a new file in the `models/` directory (e.g., `models/my_new_model.py`).
2. Implement your model by subclassing `BaseFairseqModel`.
3. Register your model and architecture with the `@register_model` and `@register_model_architecture` decorators.

Example:
```python
@register_model("my_new_model")
class MyNewModel(BaseFairseqModel):
    # Model implementation
    ...

@register_model_architecture("my_new_model", "my_new_model_arch")
def base_architecture(args):
    # Set default hyperparameters
    ...
```

### Adding a New Task

1. Create a new file in the `tasks/` directory (e.g., `tasks/my_new_task.py`).
2. Implement your task by subclassing `FairseqTask`.
3. Register your task with the `@register_task` decorator.

Example:
```python
@register_task("my_new_task")
class MyNewTask(FairseqTask):
    # Task implementation
    ...
```

## Adding More Tasks

The pipeline is designed to support multiple tasks beyond SC10:

- For ImageNet-1k, implement a task similar to the speech commands task but for image classification.
- For WMT16, implement a translation task using the fairseq translation framework.

## License

MIT 