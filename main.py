import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_cleaning import clean_data
from src.feature_engineering import add_features
from src.model_train import train_anomaly_model

def run_pipeline(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = clean_data(df)
    df = add_features(df)
    df = train_anomaly_model(df, features=[
        'bytes_in', 'bytes_out', 'duration_seconds', 'avg_packet_size'
    ])
    df.to_csv(output_path, index=False)
    print(f"✅ Analysis complete. Output saved to {output_path}")
    return df

def inspect_results(df: pd.DataFrame):
    print("🔍 Anomaly Counts:\n", df['anomaly'].value_counts(), sep='\n')
    print("\n🧪 Sample Suspicious Records:\n",
        df[df['anomaly'] == 'Suspicious'].head(), sep='\n')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='bytes_in', y='bytes_out', hue='anomaly',
        data=df, palette={'Normal':'green', 'Suspicious':'red'}
    )
    plt.title('Anomaly Detection: Bytes In vs Bytes Out')
    plt.xlabel('Bytes In')
    plt.ylabel('Bytes Out')
    plt.legend(title='Status')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    base = os.path.dirname(__file__)
    inp = os.path.join(base, 'data', 'CloudWatch_Traffic_Web_Attack.csv')
    out = os.path.join(base, 'data', 'analyzed_output.csv')

    df = run_pipeline(inp, out)
    inspect_results(df)

if __name__ == "__main__":
    main()
