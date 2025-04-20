import pandas as pd
from src.data_cleaning import clean_data
from src.feature_engineering import add_features
from src.model_train import train_anomaly_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load analyzed output
df = pd.read_csv('data/analyzed_output.csv')

# Count of detected anomalies vs normal
print("🔍 Anomaly Counts:\n")
print(df['anomaly'].value_counts())

# Optional: Display suspicious samples
print("\n🧪 Sample Suspicious Records:\n")
print(df[df['anomaly'] == 'Suspicious'].head())


# ✅ Load data
df = pd.read_csv(r'C:\Users\LENOVO\cybersecurity-web-threats\data\CloudWatch_Traffic_Web_Attack.csv')

# ✅ Clean & process
df = clean_data(df)
df = add_features(df)

# ✅ Train model and predict anomalies
features = ['bytes_in', 'bytes_out', 'duration_seconds', 'avg_packet_size']
df = train_anomaly_model(df, features)

# ✅ Save result
df.to_csv(r'C:\Users\LENOVO\cybersecurity-web-threats\data\analyzed_output.csv', index=False)
print("✅ Analysis complete. Output saved to data/analyzed_output.csv")

# ✅ Visualize anomaly results
# sns.scatterplot(x='bytes_in', y='bytes_out', hue='anomaly', data=df)
# plt.title("Anomaly Detection - Bytes In vs Bytes Out")
# plt.xlabel("Bytes In")
# plt.ylabel("Bytes Out")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Scatter plot: Bytes In vs Bytes Out
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bytes_in', y='bytes_out', hue='anomaly', data=df, palette=['green', 'red'])
plt.title('📉 Anomaly Detection: Bytes In vs Bytes Out')
plt.xlabel('Bytes In')
plt.ylabel('Bytes Out')
plt.legend(title='Anomaly Status')
plt.grid(True)
plt.tight_layout()
plt.show()

