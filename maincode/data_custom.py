import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import random
from pathlib import Path

class data_custom(Dataset):
    def __init__(self, name, data_path = None):
        self.name = name
        self.data_path = data_path
        if self.name == 'train':
            random.shuffle(self.data_path)
            self.data_path = self.data_path[:12000]
        if self.name == 'val':
            random.shuffle(self.data_path)
            self.data_path = self.data_path[:800]
        

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        clean_stftpt_path, noisy_stftpt_path, lippt_path = self.data_path[idx]

       
        stft_clean = torch.load(clean_stftpt_path)
        stft_noisy = torch.load(noisy_stftpt_path)
        lippt = torch.load(lippt_path)
        return {'clean': stft_clean, 'noisy': stft_noisy, 'image': lippt}
    
       

