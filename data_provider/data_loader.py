import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        print(f"🚀 DataLoader Output - seq_x.shape: {seq_x.shape}, seq_y.shape: {seq_y.shape}")  # Debugging

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='NW', data_path='/Workspace/Users/raha@verdo.com/Data_Platform_Solution/Databricks/Notebooks/Adhoc Analyses/rasmushansen/Baseline/data.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
#        df_raw.columns = ['timestamp_cet', 'CALENDAR_week_number', 'CALENDAR_day_of_week', 'CALENDAR_is_holiday', 'CALENDAR_holiday_name', 'PRODUCTION_1YA22U001ZT00', 'PRODUCTION_1YB22U001ZT00', 'PRODUCTION_1YD22U001ZT00', 'PRODUCTION_1YC22U001ZT00', 'PRODUCTION_5XE10B001YJ01', 'CONSUMPTION_volume_m3', 'CONSUMPTION_energy_kwh']
        df_raw.columns = ['timestamp_cet', 'CALENDAR_week_number', 'CALENDAR_day_of_week', 'CALENDAR_is_holiday', 'CALENDAR_holiday_name', 'PRODUCTION_52UX01M001YQ00', 'PRODUCTION_52UX01T001YQ00', 'PRODUCTION_52UX01M002YQ00', 'PRODUCTION_1UX03T001YQ00', 'PRODUCTION_1UX03M001YQ00', 'PRODUCTION_1UX03M002YQ00', 'PRODUCTION_1YA22U001ZT00', 'PRODUCTION_1YB22U001ZT00', 'PRODUCTION_1YD22U001ZT00', 'PRODUCTION_1YC22U001ZT00', 'PRODUCTION_5XE10B001YJ01', 'CONSUMPTION_volume_m3', 'CONSUMPTION_energy_kwh']
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('timestamp_cet')
        df_raw = df_raw[['timestamp_cet'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.70)
        num_test = int(len(df_raw) * 0.15)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'NW':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'W':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp_cet']][border1:border2]
        df_stamp['timestamp_cet'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp_cet'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_CustomV2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='4', data_path='data.csv', use_weather=False,
                 target=['Hz', 'LzNord', 'LzSyd', 'Drb'], scale=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_weather = use_weather

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __preprocessing_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw["LzSyd"] = df_raw["PRODUCTION_5XE10B001YJ01"] + df_raw["PRODUCTION_1YC22U001ZT00"] #Sum LzSyd and Elkedel
        df_raw.drop(columns=["PRODUCTION_5XE10B001YJ01", "PRODUCTION_1YC22U001ZT00"], inplace=True) #Remove old LzSyd and Elkedel
        df_raw.rename(columns={"PRODUCTION_1YA22U001ZT00": "Hz"}, inplace=True)
        df_raw.rename(columns={"PRODUCTION_1YB22U001ZT00": "LzNord"}, inplace=True)
        df_raw.rename(columns={"PRODUCTION_1YD22U001ZT00": "Drb"}, inplace=True)
        print(f"Before sort: ", df_raw.columns.tolist())
        desired_order = [
            'timestamp_cet', 'CALENDAR_week_number', 'CALENDAR_day_of_week', 'CALENDAR_is_holiday', 'CALENDAR_holiday_name',
            'CONSUMPTION_volume_m3', 'CONSUMPTION_energy_kwh',
            'Hz', 'LzNord', 'LzSyd', 'Drb'
        ]
        return df_raw[desired_order]


    def __read_data__(self):
        self.scaler = StandardScaler()
        #df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        df_raw = self.__preprocessing_data__()
        print(f"After sort: ", df_raw.columns.tolist())
        
        # ✅ Base feature columns (without weather)
        base_columns = [
            'timestamp_cet', 'CALENDAR_week_number', 'CALENDAR_day_of_week', 'CALENDAR_is_holiday', 'CALENDAR_holiday_name',
            'PRODUCTION_1YA22U001ZT00', 'PRODUCTION_1YB22U001ZT00', 'PRODUCTION_1YC22U001ZT00', 'PRODUCTION_1YD22U001ZT00',
            'PRODUCTION_5XE10B001YJ01', 'CONSUMPTION_volume_m3', 'CONSUMPTION_energy_kwh'
        ]
        #Husk at summere PRODUCTION_5XE10B001YJ01 og PRODUCTION_1YC22U001ZT00
        
        # ✅ Identify weather columns (anything not in base_columns)
        all_columns = set(df_raw.columns)
        weather_columns = list(all_columns.difference(base_columns))

        # ✅ Remove target columns safely
        cols = list(df_raw.columns)
        for target_col in self.target:  # Handle multiple targets
            if target_col in cols:
                cols.remove(target_col)
        for weather_col in weather_columns:
            if weather_col in cols:
                cols.remove(weather_col)
        cols.remove('timestamp_cet')
        cols.remove('CALENDAR_holiday_name')

        # ✅ Use filtered columns for df_raw (keeping date)
        df_raw = df_raw[['timestamp_cet'] + cols + self.target]  

        # ✅ Train-Test-Validation Splitting
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        # ✅ Feature Selection (First)
        if self.features in ['M', 'MS', '4']:
            df_data = df_raw.drop(columns=['timestamp_cet'])  # Remove date but keep all other features
        elif self.features == 'S':
            df_data = df_raw[self.target]

        # ✅ Append Weather Data *Only If Needed*
        if self.use_weather and len(weather_columns) > 0:
            weather_df = df_raw[weather_columns]  # Separate weather data
            df_data = pd.concat([df_data, weather_df], axis=1)  # Append weather features

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ✅ Timestamp Encoding
        df_stamp = df_raw[['timestamp_cet']][border1:border2]
        df_stamp['timestamp_cet'] = pd.to_datetime(df_stamp.timestamp_cet)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['timestamp_cet'].dt.month
            df_stamp['day'] = df_stamp['timestamp_cet'].dt.day
            df_stamp['weekday'] = df_stamp['timestamp_cet'].dt.weekday
            df_stamp['hour'] = df_stamp['timestamp_cet'].dt.hour
            data_stamp = df_stamp.drop(columns=['timestamp_cet']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp_cet'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # ✅ Store Processed Data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #print(f"🚀 DataLoader Output - seq_x.shape: {seq_x.shape}, seq_y.shape: {seq_y.shape}")  # Debugging
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    # def __init__(self, root_path, flag='train', size=None,
    #              features='NW', data_path='data.csv',
    #              target=['PRODUCTION_1YA22U001ZT00', 'PRODUCTION_1YB22U001ZT00', 'PRODUCTION_1YC22U001ZT00', 'PRODUCTION_1YD22U001ZT00'],
    #              scale=True, timeenc=0, freq='h'):
    #     if size is None:
    #         self.seq_len = 24 * 4 * 4
    #         self.label_len = 24 * 4
    #         self.pred_len = 24 * 4
    #     else:
    #         self.seq_len, self.label_len, self.pred_len = size

    #     assert flag in ['train', 'test', 'val']
    #     type_map = {'train': 0, 'val': 1, 'test': 2}
    #     self.set_type = type_map[flag]

    #     self.features = features
    #     self.target = target
    #     self.scale = scale
    #     self.timeenc = timeenc
    #     self.freq = freq

    #     self.root_path = root_path
    #     self.data_path = data_path
    #     self.__read_data__()

    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['timestamp_cet'])

    #     df_raw['CALENDAR_holiday_name'].fillna('None', inplace=True)

    #     target_columns = self.target
    #     weather_columns = list(df_raw.columns)
    #     feature_columns_list = ['timestamp_cet', 'CALENDAR_week_number', 'CALENDAR_day_of_week', 'CALENDAR_is_holiday', 'CALENDAR_holiday_name', 'PRODUCTION_1YA22U001ZT00', 'PRODUCTION_1YB22U001ZT00', 'PRODUCTION_1YD22U001ZT00', 'PRODUCTION_1YC22U001ZT00', 'PRODUCTION_5XE10B001YJ01', 'CONSUMPTION_volume_m3', 'CONSUMPTION_energy_kwh']
    #     for col in feature_columns_list:
    #         weather_columns.remove(col)
    #     feature_columns = list(df_raw.columns)
    #     for col in weather_columns:
    #         feature_columns.remove(col)
    #     feature_columns.remove('timestamp_cet')
    #     feature_columns.remove('CALENDAR_holiday_name')
    #     #for col in target_columns:
    #     #    feature_columns.remove(col)
    #     if self.features == 'W':
    #         selected_columns = feature_columns + weather_columns
    #     elif self.features == 'NW':
    #         selected_columns = feature_columns
    #     else:
    #         print(self.features)
    #         raise ValueError("Invalid feature mode. Use 'W' for Weather or 'NW' for No Weather")
    #     print(f"Number of input features: {len(feature_columns)}")

    #     num_train = int(len(df_raw) * 0.75)
    #     num_test = int(len(df_raw) * 0.15)
    #     num_vali = len(df_raw) - num_train - num_test

    #     border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
    #     border2s = [num_train, num_train + num_vali, len(df_raw)]
    #     border1, border2 = border1s[self.set_type], border2s[self.set_type]

    #     df_features = df_raw[feature_columns]
    #     df_targets = df_raw[target_columns]

    #     if self.scale:
    #         train_data = df_features.iloc[border1s[0]:border2s[0]]
    #         self.scaler.fit(train_data)
    #         data_x = self.scaler.transform(df_features)
    #         data_y = df_targets.values
    #     else:
    #         data_x = df_features.values
    #         data_y = df_targets.values

    #     df_stamp = df_raw[['timestamp_cet']][border1:border2]
    #     df_stamp['timestamp_cet'] = pd.to_datetime(df_stamp.timestamp_cet)
    #     if self.timeenc == 0:
    #         df_stamp['month'] = df_stamp['timestamp_cet'].dt.month
    #         df_stamp['day'] = df_stamp['timestamp_cet'].dt.day
    #         df_stamp['weekday'] = df_stamp['timestamp_cet'].dt.weekday
    #         df_stamp['hour'] = df_stamp['timestamp_cet'].dt.hour
    #         data_stamp = df_stamp.drop(['timestamp_cet'], axis=1).values
    #     elif self.timeenc == 1:
    #         data_stamp = time_features(pd.to_datetime(df_stamp['timestamp_cet'].values), freq=self.freq)
    #         data_stamp = data_stamp.transpose(1, 0)

    #     self.data_x = data_x[border1:border2]
    #     self.data_y = data_y[border1:border2]
    #     self.data_stamp = data_stamp
    #     print(f"data_y.shape: {self.data_y.shape}")

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark

    # def __len__(self):
    #     return len(self.data_x) - self.seq_len - self.pred_len + 1

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
