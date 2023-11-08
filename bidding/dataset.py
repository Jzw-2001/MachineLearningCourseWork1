import os
import sys
import json
import random
import pandas as pd
import numpy as np
from torch.utils import data
import torch

class ShillBiddingDataset(data.Dataset):
    def __init__(self, file_dir):

        # file_dir represents the path to the csv file
        super(ShillBiddingDataset, self).__init__()

        with open(file_dir, 'r') as f:
#             Record_ID,Auction_ID,Bidder_ID,Bidder_Tendency,Bidding_Ratio,Successive_Outbidding,Last_Bidding,Auction_Bids,Starting_Price_Average,Early_Bidding,Winning_Ratio,Auction_Duration,Class
            # 1,732,_***i,0.2,0.4,0,0.0000277778,0,0.993592814,0.0000277778,0.666666667,5,0
            # 2,732,g***r,0.024390244,0.2,0,0.0131226852,0,0.993592814,0.0131226852,0.944444444,5,0
            # 3,732,t***p,0.142857143,0.2,0,0.0030416667,0,0.993592814,0.0030416667,1,5,0
            # 4,732,7***n,0.1,0.2,0,0.0974768519,0,0.993592814,0.0974768519,1,5,0
            self.data = pd.read_csv(f)
            self.class_label = self.data['Class']
            self.bidder_tendency = self.data['Bidder_Tendency']
            self.bidding_ratio = self.data['Bidding_Ratio']
            self.successive_outbidding = self.data['Successive_Outbidding']
            self.last_bidding = self.data['Last_Bidding']
            self.auction_bids = self.data['Auction_Bids']
            self.starting_price_average = self.data['Starting_Price_Average']
            self.early_bidding = self.data['Early_Bidding']
            self.winning_ratio = self.data['Winning_Ratio']
            self.auction_duration = self.data['Auction_Duration']

        self.shuffle_idx = list(range(len(self.data)))
        random.shuffle(self.shuffle_idx)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index = self.shuffle_idx[index]
        
        features = torch.tensor([
            self.bidder_tendency[index],
            self.bidding_ratio[index],
            self.successive_outbidding[index],
            self.last_bidding[index],
            self.auction_bids[index],
            self.starting_price_average[index],
            self.early_bidding[index],
            self.winning_ratio[index],
            self.auction_duration[index]
        ], dtype=torch.float32)

        label = torch.tensor(self.class_label[index], dtype=torch.long)

        return features, label