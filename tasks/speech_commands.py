# tasks/speech_commands.py
import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset
import numpy as np
from fairseq.criterions import CRITERION_REGISTRY
from omegaconf import OmegaConf
import torch.nn.functional as F

@register_task('speech_commands')
class SpeechCommandsTask(FairseqTask):
    """
    A Fairseq Task for classifying audio commands.
    Supports both 10-class and 35-class configurations.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data', type=str, help='Path to the SpeechCommands dataset root')
        parser.add_argument('--classification-head-name', type=str, default='speech_commands_head', help='Name for the classification head')
        parser.add_argument('--num-classes', type=int, default=10, help='Number of target classes')
        parser.add_argument('--max-audio-frames', type=int, default=16000,
                            help='Max number of audio samples or frames in input (for raw waveforms)')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        Called by Fairseq to create the task instance.
        """
        if not os.path.isdir(args.data):
            raise FileNotFoundError(f"SpeechCommands data folder not found: {args.data}")
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_classes = args.num_classes

        # Define class labels
        # 10-class subset (common commands)
        self.labels_10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        
        # Full 35-class set (all commands)
        self.labels_35 = [
            "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
            "backward", "forward", "follow", "learn", "visual"
        ]

        # Select labels based on num_classes
        if self.num_classes == 10:
            self.labels = self.labels_10
        elif self.num_classes == 35:
            self.labels = self.labels_35
        else:
            raise ValueError(f"Number of classes must be 10 or 35, got {self.num_classes}")

        # Define label mapping
        self.label_to_index = {lab: i for i, lab in enumerate(self.labels)}
        self.datasets = {}

    def load_dataset(self, split: str, **kwargs):
        """
        Loads the dataset for train/valid/test splits.
        Fairseq calls this method. We'll define a custom Dataset.
        """
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"Unknown split {split}. Must be one of: train, valid, test.")

        # Use our dataset wrapper for SpeechCommands
        dataset = SpeechCommandsWrapper(
            root=self.args.data,
            subset=split,
            label_list=self.labels,
            max_audio_frames=self.args.max_audio_frames,
            debug=self.args.debug
        )
        
        # Set device based on CUDA availability
        if torch.cuda.is_available():
            dataset.set_device(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            dataset.set_device(torch.device("mps"))

        self.datasets[split] = dataset

    def dataset(self, split):
        """
        Return the loaded dataset for a given split.
        """
        return self.datasets[split]

    def build_model(self, args):
        """
        Build the model architecture (with classification head).
        """
        model = super().build_model(args)
        # Register classification head for speech commands
        model.register_classification_head('speech_commands_head', num_classes=self.num_classes)
        return model

    def max_positions(self):
        """
        Return the maximum input length supported by the model.
        """
        return (self.args.max_audio_frames, )

    def build_criterion(self, cfg):
        """
        Build the criterion for the task.
        """
        from fairseq.criterions import CRITERION_REGISTRY
        # Create a new config with the required fields
        criterion_cfg = OmegaConf.create({
            "criterion": "cross_entropy",
            "sentence_avg": True,
            "report_accuracy": True
        })
        return CRITERION_REGISTRY['cross_entropy'](criterion_cfg, self)
        
    def compute_metrics(self, logits, targets):
        """
        Compute metrics for the task.
        """
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate loss
        loss = F.cross_entropy(logits, targets)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'ntokens': total,
            'nsentences': total,
            'sample_size': total
        }

class SpeechCommandsWrapper(FairseqDataset):
    """
    A wrapper around torchaudio's SPEECHCOMMANDS that properly uses the official dataset split.
    Uses the official validation_list.txt and testing_list.txt files for splits.
    """

    def __init__(self, root, subset, label_list, max_audio_frames=16000, debug=False):
        super().__init__()
        self.root = root
        self.subset = subset
        self.labels = label_list
        self.max_audio_frames = max_audio_frames
        self.debug = debug
        self.device = None  # Will be set by the task
        
        # Read validation and test lists
        val_list_path = os.path.join(root, 'validation_list.txt')
        test_list_path = os.path.join(root, 'testing_list.txt')
        
        validation_files = set()
        testing_files = set()
        
        if os.path.exists(val_list_path):
            with open(val_list_path, 'r') as f:
                validation_files = set(line.strip() for line in f)
        else:
            raise FileNotFoundError(f"Validation list not found at {val_list_path}")
            
        if os.path.exists(test_list_path):
            with open(test_list_path, 'r') as f:
                testing_files = set(line.strip() for line in f)
        else:
            raise FileNotFoundError(f"Testing list not found at {test_list_path}")
            
        # Load the samples based on the current split
        self.samples = []
        self.sizes = []  # Store sizes for each sample
        
        # Process each directory of interest (each label)
        for label in self.labels:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                print(f"Warning: Label directory {label_dir} not found. Skipping.")
                continue
                
            # Process each audio file in the label directory
            for filename in os.listdir(label_dir):
                if not filename.endswith('.wav'):
                    continue
                    
                file_path = os.path.join(label, filename)  # Relative path as stored in list files
                
                # Determine which split this file belongs to
                is_val = file_path in validation_files
                is_test = file_path in testing_files
                
                # Only add files that belong to the requested split
                if (self.subset == 'valid' and is_val) or \
                   (self.subset == 'test' and is_test) or \
                   (self.subset == 'train' and not is_val and not is_test):
                    
                    # Load the audio file
                    audio_path = os.path.join(root, file_path)
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Ensure consistent sample rate
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
                    
                    # Ensure expected shape and length
                    waveform = waveform.squeeze(0)  # Convert from [1, T] to [T]
                    
                    # Store size (length) of the audio
                    self.sizes.append(waveform.size(0))
                    
                    # Add to samples collection
                    self.samples.append((waveform, self.labels.index(label)))
                    
                    # In debug mode, limit the number of samples
                    if self.debug and len(self.samples) >= 100:
                        break
            
            # In debug mode, limit the number of labels
            if self.debug and len(self.samples) >= 100:
                break

        # Convert sizes to numpy array for faster access
        self.sizes = np.array(self.sizes)
        print(f"Loaded {len(self.samples)} samples for {subset} split")
        
    def set_device(self, device):
        """Set the device for the dataset"""
        self.device = device
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        waveform, label = self.samples[index]
        # Move tensors to the same device as the model if needed
        if self.device is not None:
            waveform = waveform.to(self.device)
        return waveform, label

    def collater(self, samples):
        """
        Merge list of (waveform, class_idx) pairs into a batch.
        """
        if len(samples) == 0:
            return {}
        # Sort by length descending
        samples = sorted(samples, key=lambda x: x[0].size(0), reverse=True)
        lengths = [s[0].size(0) for s in samples]
        max_len = max(lengths)
        batch_wave = torch.zeros(len(samples), max_len)
        batch_label = torch.zeros(len(samples), dtype=torch.long)
        for i, (wave, lbl) in enumerate(samples):
            batch_wave[i, :wave.size(0)] = wave
            batch_label[i] = lbl
        
        # Calculate total number of tokens (audio frames)
        ntokens = sum(lengths)
        
        # Create the sample dictionary in the format expected by Fairseq
        sample = {
            'id': torch.arange(len(samples)),
            'net_input': {
                'src_tokens': batch_wave,    # shape [B, T]
                'src_lengths': torch.tensor(lengths, dtype=torch.long)
            },
            'target': batch_label,
            'ntokens': ntokens,  # Add this field for the criterion
            'nsentences': len(samples)  # Add this field for the criterion
        }
        
        return sample

    def size(self, index):
        """Return an example's size as a float or tuple."""
        return self.sizes[index]

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to enforce max-tokens during batching."""
        return self.sizes[index]
        
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based on this order."""
        # Sort by audio length for efficient batching
        return np.argsort(self.sizes)
        
    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
        
    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer than specified in max_sizes."""
        if isinstance(max_sizes, int):
            max_sizes = [max_sizes]
            
        if not isinstance(max_sizes, (list, tuple)):
            return indices, []
            
        ignored = []
        for idx in indices.tolist():
            size = self.size(idx)
            if size > max_sizes[0]:
                ignored.append(idx)
                
        if len(ignored) > 0:
            indices = np.array([idx for idx in indices.tolist() if idx not in ignored])
            
        return indices, ignored
