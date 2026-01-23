import shutil
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS 
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
from utils.audio_proc import fix_audio_length
from torchaudio.datasets.utils import _extract_tar, _load_waveform
from tqdm import tqdm
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import logging

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "noise"]
DATA_DIR = "data/SpeechCommands"
SETS = ["train", "valid", "test"]
url = "speech_commands_v0.02"

class SCLoader:
    def __init__(self, batch_size:int, num_workers: int, feature, sample_rate:int):
        self.name = 'SC'
        self.batch_size = batch_size
        self.labels = LABELS
        self.duration = 1.0 
        self.sr = sample_rate
        self.out_dim = 10
        self.data_dir = Path(DATA_DIR)
        self.feature = feature

        # Check if data is prepared
        data_subdir, archive_file  = self.data_dir / url, self.data_dir / (url+".tar.gz")
        if not archive_file.exists():
            raise RuntimeError(
                    f"The file {archive_file} doesn't exist. "
                    "Please check the ``root`` path or run `dvc pull` to download it"
                    )
        if not data_subdir.exists(): 
            _extract_tar(archive_file, self.data_dir)

        trainset = SubsetSC(self.data_dir.parent, "training", self.feature, self.sr)
        testset = SubsetSC(self.data_dir.parent, "testing", self.feature, self.sr)
        validset = SubsetSC(self.data_dir.parent, "validation", self.feature, self.sr)

        tensor_missing = sum([(not (self.data_dir/f"{dset}_tensors.pt").exists() 
                               or not (self.data_dir/f"{dset}_labels.pt").exists())
                              for dset in SETS])
        if tensor_missing:
               self.get_tensors([trainset, validset, testset])

        train_tensors, train_labels = (torch.load(self.data_dir/"train_tensors.pt"), 
                                       torch.load(self.data_dir/"train_labels.pt"))
        valid_tensors, valid_labels = (torch.load(self.data_dir/"valid_tensors.pt"), 
                                       torch.load(self.data_dir/"valid_labels.pt"))
        test_tensors, test_labels = (torch.load(self.data_dir/"test_tensors.pt"), 
                                       torch.load(self.data_dir/"test_labels.pt"))

        trains = TensorDataset(train_tensors, train_labels)
        valids = TensorDataset(valid_tensors, valid_labels)
        tests = TensorDataset(test_tensors, test_labels)

        self.train = DataLoader(trains, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(valids, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(tests, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size
        self.labels = LABELS
        self.out_dim = 10
        if self.feature:
            out_shape = self.feature(torch.randn(1, int(self.sr*self.duration))).shape
            self.in_chan, self.in_size = out_shape[0], tuple(out_shape[1:])
            logging.info(f"Feature size: {self.in_size}")
        else:
            self.in_chan, self.in_size = 1, (1, int(self.sr*self.duration))
            logging.info("Raw data")

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

    def get_config(self):
        return {
                "task": self.name,
                "sample_rate": self.sr,
                "in_chan": self.in_chan,
                "in_size": self.in_size,
                "out_dim": self.out_dim,
                "train_samples": len(self.train.dataset),
                "valid_samples": len(self.valid.dataset),
                "test_samples": len(self.test.dataset),
                "feature": str(self.feature).split('(')[0],
                }

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dataset_dir: Path, subset: str = "training", feature=None, 
                 sample_rate: int = 16000):
        super().__init__(dataset_dir, download=False, subset=subset)
        self.labels = LABELS
        self.sr = sample_rate
        self.duration = 1.0
        self.resampler = torchaudio.transforms.Resample(16000, self.sr)
        self._walker = [file for file in self._walker if file.split('/')[-2] in LABELS] 
        self.subset = subset
        self.feature = feature

    def __len__(self):
        return int(len(self._walker))

    def label_to_target(self, word):
        return torch.tensor(LABELS.index(word))

    def __getitem__(self, idx):
        metadata = self.get_metadata(idx)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        waveform = self.resampler(waveform)
        waveform = fix_audio_length(waveform, t=self.duration, sr=self.sr)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-10)
        feat = self.feature(waveform)
        target = self.label_to_target(metadata[2])
        return feat, target
