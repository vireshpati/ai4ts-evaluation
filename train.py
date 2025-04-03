import os
import subprocess
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a speech commands classification model")
    
    # Add command-line arguments
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
    
    # Use provided data directory or default to data/speech_commands
    data_dir = args.data_dir if args.data_dir else os.path.join(pipeline_dir, "data", "speech_commands")
    
    # Verify data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Using data from: {data_dir}")
    
    # Determine device
    if torch.cuda.is_available():
        print("Using CUDA GPU for training")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU) for training")
        # Set environment variable for MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        raise RuntimeError("No GPU available. This task requires GPU acceleration.")
    
    # Form the fairseq-train command
    cmd = [
        "fairseq-train",
        "--user-dir", user_dir,
        "--task", "speech_commands",
        "--data", data_dir,
        "--arch", "simple_linear_transformer_arch",
        "--criterion", "cross_entropy",
        "--classification-head-name", "speech_commands_head",
        "--optimizer", "adam",
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--max-epoch", str(args.max_epochs),
        "--num-classes", str(args.num_classes),
        "--max-audio-frames", "16000",  # 1 second of audio at 16kHz
        "--save-dir", os.path.join(pipeline_dir, f"checkpoints_speech_{args.num_classes}class"),
        "--valid-subset", "valid",
        "--save-interval", "1",
        "--log-interval", "10",
    ]
    
    # Add debug flag if requested
    if args.debug:
        cmd.append("--debug")
        print("Running in debug mode with smaller dataset")
    
    print(f"Training Speech Commands classifier with {args.num_classes} classes")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()