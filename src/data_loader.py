

import torch
from torch.utils.data import Dataset

import pandas as pd
import nibabel as nib
import numpy as np

class DataGenerator(Dataset):
    
    
    def __init__(self, path, data_col="9dof_2mm_vol"):
        data = pd.read_csv(path)
        self.eids = data["eid"].values.tolist()
        self.age = data["age_at_scan"].values.tolist()
        self.vol_paths = data[data_col].values.tolist()
        self.scans = {k: v for k, v in zip(self.eids, self.vol_paths)}
        self.length = len(self.scans)
        self.data_col = data_col
        self.input_dim = {
            "9dof_1mm_vol": (1, 182, 218, 182),
            "9dof_2mm_vol": (1, 91, 109, 91),
            "9dof_4mm_vol": (1, 46, 55, 46),
            "9dof_8mm_vol": (1, 23, 27, 23)
        }[self.data_col]


    def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        eids = self.eids[index]
        scan_paths_temp = self.scans[eids]
        X = np.float32(nib.load(scan_paths_temp).get_fdata())
        y = self.age[index]
        X = (X-X.mean())/X.std()

        X = X[np.newaxis,:,:,:]
        y = np.float32(np.array([y]))

        return X, y