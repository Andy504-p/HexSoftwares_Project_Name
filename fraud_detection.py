"""
Simple Fraud Detection Example Using Isolation Forest

This script simulates a financial transactions dataset,
trains an Isolation Forest model to detect anomalies (fraudulent transactions),
and outputs detected anomalies.

Requirements:
  - pandas
  - scikit-learn

Run:
  python fraud_detection_example.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def generate_sample_data(num_samples=1000):
    """
    Generates a sample financial transaction dataset.
    Columns:
      - amount: transaction amount (normal around 50, fraud outliers up to 1000)
      - transaction_type: type of txn (0=payment,1=transfer,2=withdrawal)
      - location: simulated location code (0 to 4)
      - hour_of_day: time of transaction (0-23)
    Some transactions are labelled as anomalies (frauds).
    """
    np.random.seed(42)
    amounts = np.random.normal(loc=50, scale=10, size=num_samples)
    transaction_types = np.random.choice([0, 1, 2], size=num_samples)
    locations = np.random.choice([0, 1, 2, 3, 4], size=num_samples)
    hour_of_day = np.random.choice(range(24), size=num_samples)

    # Introduce some anomalies with large amounts and odd hours
    num_frauds = int(0.02 * num_samples)
    anomaly_indices = np.random.choice(num_samples, num_frauds, replace=False)
    amounts[anomaly_indices] = np.random.uniform(300, 1000, size=num_frauds)  # large amounts
    hour_of_day[anomaly_indices] = np.random.choice([0, 1, 2, 3, 4], size=num_frauds)  # unusual hours

    data = pd.DataFrame({
        "amount": amounts,
        "transaction_type": transaction_types,
        "location": locations,
        "hour_of_day": hour_of_day
    })
    return data


def preprocess_features(df):
    """
    Preprocesses the features for model training:
    - Scale numeric features (amount, hour_of_day)
    - Encode categorical features (transaction_type, location)
    """
    df_proc = df.copy()

    # Encode categorical columns
    le_type = LabelEncoder()
    df_proc['transaction_type_enc'] = le_type.fit_transform(df_proc['transaction_type'])

    le_loc = LabelEncoder()
    df_proc['location_enc'] = le_loc.fit_transform(df_proc['location'])

    # Scaling
    scaler = StandardScaler()
    df_proc[['amount_scaled', 'hour_scaled']] = scaler.fit_transform(df_proc[['amount', 'hour_of_day']])

    features = df_proc[['amount_scaled', 'hour_scaled', 'transaction_type_enc', 'location_enc']]
    return features


def main():
    print("Generating sample transaction data...")
    data = generate_sample_data(num_samples=1000)

    print("Preprocessing features...")
    features = preprocess_features(data)

    print("Splitting data into train and test sets...")
    X_train, X_test = train_test_split(features, test_size=0.3, random_state=42)

    print("Training Isolation Forest model for anomaly detection...")
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X_train)

    print("Predicting anomalies on test set...")
    preds = model.predict(X_test)
    # -1 indicates anomalies, 1 is normal
    anomaly_mask = (preds == -1)

    # Output anomalies
    anomalies = X_test[anomaly_mask]
    anomalies_orig = data.loc[anomalies.index]

    print(f"Number of transactions flagged as anomalies/frauds: {len(anomalies)}")
    print("\nExamples of detected anomalies (fraudulent transactions):")
    print(anomalies_orig.head(10))


if __name__ == "__main__":
    main()

