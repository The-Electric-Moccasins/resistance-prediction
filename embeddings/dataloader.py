import torch
from torch.utils.data import Dataset
import numpy as np


class TheDataSet(Dataset):

    def __init__(self, datafile='data/fulldata.npy'):
        """
        Args:
            datafile (string): Path to the data file
        """
        fulldata = np.load(datafile, allow_pickle=True)
        self.X = fulldata[:, :-1]
        self.y = fulldata[:, -1]
        _, vector_len = self.X.shape


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]

    def num_features(self):
        return self.X.shape[1]
