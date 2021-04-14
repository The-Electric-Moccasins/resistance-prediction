import torch
from torch.utils.data import Dataset
import numpy as np


class TheDataSet(Dataset):

    def __init__(self, datafile='data/fulldata.npy', pad_to_360=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        fulldata = np.load(datafile)
        self.X = fulldata[:, :-1]
        self.y = fulldata[:, -1]
        _, vector_len = self.X.shape

        #
        if pad_to_360:
            self.X = np.pad(self.X, ((0, 0), (0, 360 - vector_len)), 'constant', constant_values=(0, 0))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]

    def num_features(self):
        return self.X.shape[1]
