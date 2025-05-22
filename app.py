import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_features(df):
    df_proc = df.copy()

    le_type = LabelEncoder()
    df_proc['transaction_type_enc'] = le_type.fit_transform(df_proc['transaction_type'])

    le_loc = LabelEncoder()
    df_proc['location_enc'] = le_loc.fit_transform(df_proc['location'])

    scaler = StandardScaler()
    df_proc[['amount_scaled', 'hour_scaled']] = scaler.fit_transform(df_proc[['amount', 'hour_of_day']])

    features = df_proc[['amount_scaled', 'hour_scaled', 'transaction_type_enc', 'location_enc']]
    return features


st.title("Financial Fraud Detection with Isolation Forest")

uploaded_file = st.file_uploader("Upload CSV file of transactions", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Sample:")
    st.dataframe(data.head())

    if st.button("Detect Anomalies"):
        features = preprocess_features(data)
        model = IsolationForest(contamination=0.02, random_state=42)
        model.fit(features)
        preds = model.predict(features)

        data['anomaly'] = preds
        anomalies = data[data['anomaly'] == -1]

        st.write(f"Number of anomalies detected: {len(anomalies)}")
        st.write("Anomalies:")
        st.dataframe(anomalies)
