import os
import sys
import json
import random
import pandas as pd
import numpy as np
from torch.utils import data
import torch
from datetime import datetime

def check_weekday(date_string):
    date = datetime.strptime(date_string, "%d/%m/%Y")
    weekday = date.weekday()  # 获取星期几，0代表星期一，6代表星期日
    
    if weekday < 5:
        return 1
    else:
        return 0

class SeoulBikeDataset(data.Dataset):
    def __init__(self, file_dir):
        super(SeoulBikeDataset, self).__init__()

        with open(file_dir, 'r') as f:
            self.data = pd.read_csv(f)
            # label是第一列
            # features是第二列到最后一列
            first_col = self.data.iloc[:, 0]
            self.label = first_col
            other_cols = self.data.iloc[:, 1:]
            self.features = other_cols

        self.shuffle_idx = list(range(len(self.data)))
        random.shuffle(self.shuffle_idx)
    

    
    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, index):
        index = self.shuffle_idx[index]
        # features = []
        # for i in range(len(self.features.iloc[index])):
        #     features.append(self.features.iloc[index][i])
        # print("features", features)
        # features = torch.tensor(features, dtype=torch.float32)
        features = torch.tensor(self.features.iloc[index].values, dtype=torch.float32)
        label = torch.tensor(self.label.iloc[index], dtype=torch.float32)
        return features, label