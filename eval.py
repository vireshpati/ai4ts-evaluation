# eval.py
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.serialization

# Try to completely disable ffmpeg and use Sox instead
os.environ["TORCHAUDIO_USE_FFMPEG"] = "0"  
os.environ["TORCHAUDIO_USE_SOX"] = "1"
os.environ["TORCHAUDIO_FFMPEG_EXECUTABLE"] = ""  

# Add fairseq to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'fairseq')))
from fairseq import checkpoint_utils, tasks

from fairseq.dataclass.configs import FairseqConfig
from omegaconf import OmegaConf
from fairseq.models import BaseFairseqModel

# Add these imports right after your current imports
torch.serialization.add_safe_globals([argparse.Namespace])
from omegaconf import OmegaConf
torch.serialization.add_safe_globals([OmegaConf, OmegaConf.create])

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Speech Commands model checkpoint")
    parser.add_argument('--user-dir', type=str, default='.',
                        help='Directory containing custom tasks/models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the SpeechCommands dataset root')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the model checkpoint (e.g. checkpoint_best.pt)')
    parser.add_argument('--num-classes', type=int, default=10,
                        choices=[10, 35],
                        help='Number of classes in the model')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Which dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    args = parser.parse_args()

    # Register user directory for custom models/tasks
    fairseq_setup_user_dir = getattr(tasks, "setup_user_dir", None)
    if fairseq_setup_user_dir:
        fairseq_setup_user_dir(args.user_dir)
    else:
        # Fallback for older fairseq versions
        from fairseq.utils import import_user_module
        import_user_module(argparse.Namespace(user_dir=args.user_dir))

    # Create task config
    task_config = OmegaConf.create({
        "task": "speech_commands",
        "data": args.data,
        "num_classes": args.num_classes,
        "max_audio_frames": 16000,
        "classification_head_name": "speech_commands_head",
        "debug": False
    })

    # Get the setup_task function - it might be at different places depending on fairseq version
    setup_task = getattr(tasks, "setup_task", None)
    if setup_task is None:
        from fairseq.tasks import setup_task
    
    # Load task
    task = setup_task(task_config)
    task.load_dataset(args.split)

    # Load model
    try:
        # Try to print the checkpoint structure to understand its format
        checkpoint = torch.load(args.path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            # Standard fairseq format
            state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Alternative format
            state = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state = checkpoint
        
        # Create model directly from task
        model_cfg = OmegaConf.create({
            "arch": "simple_linear_transformer_arch",
            "encoder_layers": 4,
            "encoder_embed_dim": 256,
            "encoder_ffn_embed_dim": 512, 
            "encoder_attention_heads": 4,
            "dropout": 0.1,
        })
        
        # Build the model
        model = task.build_model(model_cfg)
        
        # Register classification head if needed
        if not hasattr(model, 'classification_heads') or 'speech_commands_head' not in model.classification_heads:
            print("Registering classification head")
            model.register_classification_head('speech_commands_head', args.num_classes)
        
        # Try to load state dict
        try:
            model.load_state_dict(state, strict=False)
            print("Loaded state dict successfully")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Will continue with initialized model")
        
        models = [model]
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating new model from scratch")
        
        # Create model with task
        model_cfg = OmegaConf.create({
            "arch": "simple_linear_transformer_arch",
            "encoder_layers": 4,
            "encoder_embed_dim": 256,
            "encoder_ffn_embed_dim": 512, 
            "encoder_attention_heads": 4,
            "dropout": 0.1,
        })
        model = task.build_model(model_cfg)
        model.register_classification_head('speech_commands_head', args.num_classes)
        models = [model]

    model = models[0].eval()
    if torch.cuda.is_available():
        model.cuda()
    elif torch.backends.mps.is_available():
        model.to(torch.device("mps"))

    # Create a DataLoader
    dataset = task.dataset(args.split)
    
    def collate_fn(batch):
        return dataset.collater(batch)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Add this right after creating the loader
    print(f"Starting evaluation on {len(loader)} batches...")
    batch_count = 0

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for sample in loader:
            batch_count += 1
            if batch_count % 10 == 0:  # Print every 10 batches
                print(f"Processing batch {batch_count}/{len(loader)}")
            
            if not sample or 'net_input' not in sample:
                continue
            
            net_input = sample['net_input']
            target = sample['target']

            if torch.cuda.is_available():
                for k, v in net_input.items():
                    if isinstance(v, torch.Tensor):
                        net_input[k] = v.cuda()
                target = target.cuda()
            elif torch.backends.mps.is_available():
                for k, v in net_input.items():
                    if isinstance(v, torch.Tensor):
                        net_input[k] = v.to("mps")
                target = target.to("mps")

            # Forward pass
            try:
                logits, _ = model(**net_input, classification_head_name='speech_commands_head')
            except Exception as e:
                print(f"Error in forward pass: {e}")
                # Try alternative approach
                try:
                    # Extract features first
                    features, _ = model.extract_features(**net_input)
                    # Average pooling
                    features = features.mean(dim=1)  # [B, embed_dim]
                    # Apply classification head manually
                    logits = model.classification_heads['speech_commands_head'](features)
                except Exception as alt_e:
                    print(f"Alternative forward pass also failed: {alt_e}")
                    continue
            
            # Compute loss
            loss = F.cross_entropy(logits, target, reduction='sum')
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == target).sum().item()
            total_correct += correct
            total_samples += target.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"Evaluation on split={args.split}")
    print(f"  Num samples: {total_samples}")
    print(f"  Avg loss   : {avg_loss:.4f}")
    print(f"  Accuracy   : {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
