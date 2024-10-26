"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2

Landmarks Dataset
    Class wrapper for interfacing with the dataset of landmark images
    Usage: python dataset.py
"""

import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader



def get_train_val_test_loaders(batch_size):
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the LandmarksDataset constructor.
    """
    tr = AudioDataset("train")
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(AudioDataset("val"), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(AudioDataset("test"), batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_semantic_label

class AudioDataset(Dataset):
    """Dataset class for landmark images."""

    def __init__(self, partition):
        """Read in the necessary data from disk.
        """
        super().__init__()

        if partition not in ["train", "val", "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        self.partition = partition
        self.train_audio_path = './train/audio'
        # labels
        self.semantic_labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        # test and val data
        test = []
        val = []
        with open("./train/testing_list.txt", "r", encoding='utf-8') as tfile:
            test = tfile.read().split("\n")
        with open("./train/validation_list.txt", "r", encoding='utf-8') as tfile:
            val = tfile.read().split("\n")
        # Load in all the metadata we need from disk
        self.metadata = []
        for i, label in enumerate(self.semantic_labels):
            print(label)
            waves = [f for f in os.listdir(self.train_audio_path + '/'+ label) if f.endswith('.wav')]
            for wav in waves:
                fname = label + "/" + wav
                part = "train"
                if fname in test:
                    part = "test"
                if fname in val:
                    part = "val"
                self.metadata.append({
                    "filename": fname,
                    "numeric_label": i,
                    "label": label,
                    "partition": part
                })
        
        self.X, self.y = self._load_data()

    def __len__(self):
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return (image, label) pair at index `idx` of dataset."""
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """Load a single data partition from file."""
        print("loading %s..." % self.partition)
        df = [row for row in self.metadata if row["partition"] == self.partition]
        X, y = [], []
        for row in df:
            samples, _ = librosa.load(self.train_audio_path + '/' + row["filename"], sr=8000)
            if len(samples) == 8000:
                X.append(samples.reshape(1, 8000))
                y.append(row["numeric_label"])
        X = np.array(X)
        print(X.shape)
        return X, np.array(y)

    def get_semantic_label(self, numeric_label):
        """Return the string representation of the numeric class label.
        """
        return self.semantic_labels[numeric_label]
