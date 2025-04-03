import os
import torch
import torchaudio
import torch.nn.functional as F
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset
from fairseq import metrics
import numpy as np
from omegaconf import OmegaConf

@register_task('speech_commands')
class SpeechCommandsTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data', type=str, help='Path to the SpeechCommands dataset root')
        parser.add_argument('--classification-head-name', type=str, default='speech_commands_head',
                            help='Name for the classification head')
        parser.add_argument('--num-classes', type=int, default=10, help='Number of target classes')
        parser.add_argument('--max-audio-frames', type=int, default=16000,
                            help='Max number of audio samples in input (for raw waveforms)')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    @classmethod
    def setup_task(cls, args, **kwargs):
        if not os.path.isdir(args.data):
            raise FileNotFoundError(f"SpeechCommands data folder not found: {args.data}")
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_classes = args.num_classes

        # Define class labels
        self.labels_10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.labels_35 = [
            "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
            "backward", "forward", "follow", "learn", "visual"
        ]

        if self.num_classes == 10:
            self.labels = self.labels_10
        elif self.num_classes == 35:
            self.labels = self.labels_35
        else:
            raise ValueError(f"Number of classes must be 10 or 35, got {self.num_classes}")

        self.label_to_index = {lab: i for i, lab in enumerate(self.labels)}
        self.datasets = {}

    def load_dataset(self, split: str, **kwargs):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"Unknown split {split}. Must be one of: train, valid, test.")

        dataset = SpeechCommandsWrapper(
            root=self.args.data,
            subset=split,
            label_list=self.labels,
            max_audio_frames=self.args.max_audio_frames,
            debug=self.args.debug
        )

        # Move dataset to GPU if available
        if torch.cuda.is_available():
            dataset.set_device(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            dataset.set_device(torch.device("mps"))

        self.datasets[split] = dataset

    def dataset(self, split):
        return self.datasets[split]

    def build_model(self, args):
        model = super().build_model(args)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.num_classes
        )
        return model

    def max_positions(self):
        return (self.args.max_audio_frames, )

    def build_criterion(self, cfg):
        from fairseq.criterions import CRITERION_REGISTRY
        # Use cross_entropy with "report_accuracy"
        criterion_cfg = OmegaConf.create({
            "criterion": "cross_entropy",
            "sentence_avg": True,
            "report_accuracy": True
        })
        return CRITERION_REGISTRY['cross_entropy'](criterion_cfg, self)

    def valid_step(self, sample, model, criterion):
        if not sample:
            return 0, 0, {}

        # Standard cross-entropy loss
        loss, sample_size, logging_output = criterion(model, sample)

        # Compute batch accuracy
        with torch.no_grad():
            net_output = model(
                **sample["net_input"],
                classification_head_name=self.args.classification_head_name
            )
            logits = net_output[0]
            targets = sample["target"]
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            total = targets.numel()

        logging_output["n_correct"] = correct
        logging_output["n_total"] = total

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, criterion=None):
        """
        Called after validating all batches to aggregate logs (e.g., sum correct/total).
        The second argument is the same `criterion` used in valid_step.
        """

        # 1) First, let CrossEntropyCriterion log "loss", "nll_loss", etc.
        if hasattr(criterion, "reduce_metrics"):
            criterion.__class__.reduce_metrics(logging_outputs)

        # 2) Sum "n_correct" and "n_total" across all GPUs
        metrics.log_scalar_sum("n_correct", sum(log.get("n_correct", 0) for log in logging_outputs))
        metrics.log_scalar_sum("n_total",   sum(log.get("n_total", 0)   for log in logging_outputs))

        # 3) Define derived metric for accuracy
        def compute_accuracy(meters):
            correct = meters["n_correct"].sum
            total = meters["n_total"].sum
            return float(correct) / float(total) if total > 0 else 0

        metrics.log_derived("accuracy", compute_accuracy)


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
        self.device = None

        # Read official validation and test lists
        val_list_path = os.path.join(root, 'validation_list.txt')
        test_list_path = os.path.join(root, 'testing_list.txt')

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

        self.samples = []
        self.sizes = []

        # Populate samples by scanning each label folder
        for label in self.labels:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                print(f"Warning: Label directory {label_dir} not found. Skipping.")
                continue

            for filename in os.listdir(label_dir):
                if not filename.endswith('.wav'):
                    continue

                file_path = os.path.join(label, filename)  # relative path

                # Check which split it belongs to
                is_val = file_path in validation_files
                is_test = file_path in testing_files

                if ((self.subset == 'valid' and is_val) or
                    (self.subset == 'test'  and is_test) or
                    (self.subset == 'train' and not is_val and not is_test)):

                    audio_path = os.path.join(root, file_path)
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(
                            waveform, orig_freq=sample_rate, new_freq=16000
                        )
                    
                    waveform = waveform.squeeze(0)  # [T]

                    self.sizes.append(waveform.size(0))
                    self.samples.append((waveform, self.labels.index(label)))

                    # In debug mode, limit total samples
                    if self.debug and len(self.samples) >= 100:
                        break
            if self.debug and len(self.samples) >= 100:
                break

        self.sizes = np.array(self.sizes)
        print(f"Loaded {len(self.samples)} samples for {subset} split")

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        waveform, label = self.samples[index]
        if self.device is not None:
            waveform = waveform.to(self.device)
        return waveform, label

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        samples = sorted(samples, key=lambda x: x[0].size(0), reverse=True)
        lengths = [s[0].size(0) for s in samples]
        max_len = max(lengths)
        batch_wave = torch.zeros(len(samples), max_len)
        batch_label = torch.zeros(len(samples), dtype=torch.long)
        for i, (wave, lbl) in enumerate(samples):
            batch_wave[i, :wave.size(0)] = wave
            batch_label[i] = lbl

        ntokens = sum(lengths)
        sample = {
            'id': torch.arange(len(samples)),
            'net_input': {
                'src_tokens': batch_wave,
                'src_lengths': torch.tensor(lengths, dtype=torch.long),
            },
            'target': batch_label,
            'ntokens': ntokens,
            'nsentences': len(samples),
        }
        return sample

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        return np.argsort(self.sizes)

    @property
    def supports_prefetch(self):
        return False
