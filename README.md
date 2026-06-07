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