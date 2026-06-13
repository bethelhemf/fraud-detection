# Adey Innovations - Fraud Detection System

## Overview
This project aims to build a robust fraud detection system for Adey Innovations Inc. to identify fraudulent activities in e-commerce and bank credit card transactions. 

The system handles two distinct data streams:
1. **E-commerce Transactions:** Rich in behavioral context (IP, device, time).
2. **Bank Credit Transactions:** Anonymized features via PCA.

## Project Structure
```text
fraud-detection/
├── data/                       # Dataset storage (Raw and Processed)
├── notebooks/                  # Step-by-step EDA and Modeling
├── src/                        # Source code for production scripts
├── tests/                      # Unit tests
├── models/                     # Saved model artifacts (.pkl files)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

# Task 3: Model Explainability & Business Recommendations

## 1. Feature Importance Interpretation
Comparing the built-in Gini importance and SHAP summary plots, we identified the **Top 5 Drivers of Fraud**:
1. **Time Since Signup:** The most powerful driver. Low values (immediate purchase) strongly correlate with fraud.
2. **Device Velocity:** Transactions where the device ID is shared by multiple users show high SHAP values.
3. **Purchase Value:** Mid-to-high values exhibit specific risk profiles compared to legitimate patterns.
4. **Hour of Day:** Fraud spikes during late-night hours (e.g., 2 AM - 5 AM).
5. **Country Risk:** Specific country-encoded values provide a secondary uplift in fraud probability.

## 2. SHAP Force Plot Analysis
- **True Positive:** The model flagged a transaction because `time_since_signup` was < 1 minute and `device_velocity` was > 5.
- **False Positive (False Alarm):** A legitimate user was flagged because they purchased immediately after signup on a shared network (college campus), mimicking bot behavior.
- **False Negative (Missed Fraud):** The model missed this fraud because the transaction occurred on a unique device and the user had signed up 2 days prior, successfully evading the "instant purchase" rule.

## 3. Actionable Business Recommendations
Based on SHAP insights, I recommend the following for Adey Innovations:
1. **The "Instant-Buy" Friction:** Implement a mandatory 15-minute verification delay for any purchase made within 60 minutes of account creation. SHAP shows `time_since_signup` is the primary fraud signal.
2. **Device ID Blacklisting:** Any Device ID showing more than 3 unique users within a 24-hour window (`device_velocity`) should trigger a multi-factor authentication (MFA) challenge.
3. **Night-Watch Velocity Rules:** Scale up risk scoring sensitivity during the hours of 12 AM to 6 AM (local time). Our SHAP analysis shows temporal clusters for bot-driven fraudulent activities.