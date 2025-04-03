# train.py
import os
import argparse
import torch
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Train a speech commands classification model")
    
    parser.add_argument('--num-classes', type=int, default=10, choices=[10, 35],
                        help='Number of classes to use (10 or 35)')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory, if not using default data/speech_commands')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug mode with smaller dataset for faster testing')
    
    args = parser.parse_args()
    
    pipeline_dir = os.path.dirname(os.path.realpath(__file__))
    user_dir = pipeline_dir
    
    data_dir = args.data_dir if args.data_dir else os.path.join(pipeline_dir, "data", "speech_commands")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Device check
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        raise RuntimeError("No GPU/MPS available. This task is GPU-intensive.")
    
    save_dir = os.path.join(pipeline_dir, f"checkpoints_speech_{args.num_classes}class")
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        "fairseq-train",
        "--user-dir", user_dir,                  # to find custom tasks/models
        "--task", "speech_commands",
        "--data", data_dir,
        "--arch", "simple_linear_transformer_arch",  # or "vgg_transformer_sc10"
        "--criterion", "cross_entropy",
        "--classification-head-name", "speech_commands_head",
        "--optimizer", "adam",
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--max-epoch", str(args.max_epochs),
        "--num-classes", str(args.num_classes),
        "--max-audio-frames", "16000", 
        "--save-dir", save_dir,
        "--valid-subset", "valid",
        "--save-interval", "1",
        "--log-interval", "10",
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
