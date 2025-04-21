from sklearn.ensemble import IsolationForest


def train_anomaly_model(df, features):

    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[features])
    df["anomaly"] = df["anomaly"].apply(
        lambda x: "Suspicious" if x == -1 else "Normal"
    )
    return df
