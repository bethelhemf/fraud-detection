import pandas as pd
import numpy as np
import ipaddress
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudPreprocessor:
    """Explicit and reusable pipeline for Fraud Detection Preprocessing."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """Task 1: Explicit Data Cleaning."""
        logger.info("Starting Explicit Cleaning...")
        initial_rows = len(df)
        
        # 1. Handle Missing Values: Drop rows with missing crucial IDs
        df = df.dropna(subset=['user_id', 'device_id'])
        
        # 2. Handle Duplicates: Remove duplicate transactions
        df = df.drop_duplicates()
        
        # 3. Correct Data Types
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        logger.info(f"Cleaned {initial_rows - len(df)} rows. Current shape: {df.shape}")
        return df

    def engineer_features(self, df):
        """Task 1: Explicit Feature Engineering."""
        logger.info("Engineering Behavioral Features...")
        
        # Time since signup (Behavioral pattern)
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600 # hours
        
        # Hour of Day & Day of Week (Temporal patterns)
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Transaction Velocity (Device sharing)
        df['device_velocity'] = df.groupby('device_id')['user_id'].transform('count')
        
        return df

    def encode_categories(self, df, is_train=True):
        """Task 1: Professional Categorical Encoding."""
        logger.info("Encoding Categorical Variables...")
        cat_cols = ['source', 'browser', 'sex', 'country']
        
        for col in cat_cols:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # For test set, handle unseen labels by mapping to a default
                    le = self.label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        return df