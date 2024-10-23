from torch.utils.data import Dataset
import os 
import pandas as pd

class TrailDataset(Dataset):
    def __init__(self, trail_data, transform=None, target_transform=None):
        self.dataset = pd.read_csv(trail_data, sep="|")
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input = self.dataset.iloc[idx,0]
        output = self.dataset.iloc[idx,1]
        return input, output
    

