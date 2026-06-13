
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FraudPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def clean_data(self, df):
        df = df.dropna(subset=['user_id', 'device_id']).drop_duplicates()
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        return df

    def engineer_features(self, df):
        df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 60
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['user_per_device'] = df.groupby('device_id')['user_id'].transform('count')
        return df

    def encode_and_scale(self, df, is_train=True):
        cat_cols = ['source', 'browser', 'sex', 'country']
        for col in cat_cols:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        return df
