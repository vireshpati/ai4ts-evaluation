# pipeline/evaluate.py
import sys
import os
import subprocess

def main():
    """
    Minimal example to invoke fairseq-validate on the test split for speech commands.
    """
    pipeline_dir = os.path.dirname(os.path.realpath(__file__))
    user_dir = pipeline_dir

    # We'll assume the user wants to evaluate on the 'test' split
    checkpoint_path = os.path.join(pipeline_dir, "checkpoints_speech", "checkpoint_best.pt")

    cmd = [
        "fairseq-validate",
        os.path.join(pipeline_dir, "data", "speech_commands"),
        "--user-dir", user_dir,
        "--task", "speech_commands",
        "--path", checkpoint_path,
        "--criterion", "cross_entropy",
        "--batch-size", "32",
        "--valid-subset", "test",
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
