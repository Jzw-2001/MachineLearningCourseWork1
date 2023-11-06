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

        # file_dir represents the path to the csv file
        super(SeoulBikeDataset, self).__init__()


        with open(file_dir, 'r') as f:
            self.data = pd.read_csv(f)
            self.rented_count = self.data['RentedBikeCount']

            self.date = self.data['Date']
            self.hour = self.data['Hour']
            self.temperature = self.data['Temperature']
            self.humidity = self.data['Humidity']
            self.wind_speed = self.data['Windspeed']
            self.visibility = self.data['Visibility']
            self.dew_point_temperature = self.data['DewPointTemperature']
            self.solar_radiation = self.data['SolarRadiation']
            self.rainfall = self.data['Rainfall']
            self.snowfall = self.data['Snowfall']
            self.season = self.data['Seasons']
            self.holiday = self.data['Holiday']
            self.functioning_day = self.data['FunctioningDay']
            self.is_weekday = self.date.apply(check_weekday)

        from utils import SEASON2NUM, HOLIDAY2NUM, FUNCTIONINGDAY2NUM
        self.season = self.season.map(SEASON2NUM)
        self.holiday = self.holiday.map(HOLIDAY2NUM)
        self.functioning_day = self.functioning_day.map(FUNCTIONINGDAY2NUM)


    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, index):
        hour_one_hot = np.zeros(24)
        hour_one_hot[self.hour[index]] = 1
        season_one_hot = np.zeros(4)
        season_one_hot[self.season[index]] = 1
        holiday_one_hot = np.zeros(2)
        holiday_one_hot[self.holiday[index]] = 1
        functioning_day_one_hot = np.zeros(2)
        functioning_day_one_hot[self.functioning_day[index]] = 1
        is_weekday_one_hot = np.zeros(2)
        is_weekday_one_hot[self.is_weekday[index]] = 1

        features = torch.tensor([
            float(self.temperature[index]),
            float(self.humidity[index]),
            float(self.wind_speed[index]),
            float(self.visibility[index]),
            float(self.dew_point_temperature[index]),
            float(self.solar_radiation[index]),
            float(self.rainfall[index]),
            float(self.snowfall[index]),
        ], dtype=torch.float32)

        features = torch.cat((features, torch.tensor(hour_one_hot, dtype=torch.float32)))
        features = torch.cat((features, torch.tensor(season_one_hot, dtype=torch.float32)))
        features = torch.cat((features, torch.tensor(holiday_one_hot, dtype=torch.float32)))
        features = torch.cat((features, torch.tensor(functioning_day_one_hot, dtype=torch.float32)))
        features = torch.cat((features, torch.tensor(is_weekday_one_hot, dtype=torch.float32)))
        
        label = torch.tensor(self.rented_count[index], dtype=torch.float32)

        return features, label