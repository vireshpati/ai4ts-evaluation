# pipeline/train.py
import sys
import os
import subprocess

def main():
    """
    Minimal example to invoke fairseq-train on SpeechCommands using our custom code.
    For advanced usage, we can parse args ourselves or rely on the command line directly.
    """
    # Example: training on speech commands
    # We will run the fairseq CLI, specifying our user-dir, task, model, etc.
    pipeline_dir = os.path.dirname(os.path.realpath(__file__))
    user_dir = pipeline_dir  # so it can find tasks/ and models/

    cmd = [
        "fairseq-train", 
        os.path.join(pipeline_dir, "data", "speech_commands"),  # --data
        "--user-dir", user_dir,
        "--task", "speech_commands",
        "--arch", "simple_linear_transformer_arch",
        "--criterion", "cross_entropy",
        "--classification-head-name", "speech_commands_head",
        "--optimizer", "adam",
        "--lr", "0.001",
        "--batch-size", "32",
        "--max-epoch", "10",
        "--save-dir", os.path.join(pipeline_dir, "checkpoints_speech"),
        "--best-checkpoint-metric", "accuracy",
        "--maximize-best-checkpoint-metric"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
