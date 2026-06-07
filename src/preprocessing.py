import pandas as pd
import numpy as np
import ipaddress
import logging

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def robust_ip_to_int(ip):
    if isinstance(ip, (int, float, np.integer, np.floating)):
        return float(ip)
    try:
        return float(int(ipaddress.ip_address(str(ip))))
    except Exception as e:
        return 0.0

def merge_ip_to_country(fraud_df, ip_df):
    logger.info("Starting IP-Country merge...")

    fraud_df['ip_numeric'] = fraud_df['ip_address'].apply(robust_ip_to_int)
    ip_df['lower_bound_numeric'] = ip_df['lower_bound_ip_address'].astype(float)

    fraud_df = fraud_df.sort_values('ip_numeric')
    ip_df = ip_df.sort_values('lower_bound_numeric')

    merged = pd.merge_asof(
        fraud_df, ip_df, 
        left_on='ip_numeric', 
        right_on='lower_bound_numeric', 
        direction='backward'
    )

    merged.loc[merged['ip_numeric'] > merged['upper_bound_ip_address'], 'country'] = 'Unknown'
    merged['country'] = merged['country'].fillna('Unknown')

    logger.info("Merge complete.")
    return merged
