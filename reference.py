import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
from typing import Tuple

warnings.filterwarnings("ignore")

# Your existing code...

def create_correlation_heatmap(df: pd.DataFrame) -> Tuple[None, pd.DataFrame]:
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                annot_kws={"size": 12},  # Increase font size here
                cbar_kws={"shrink": .82})
    
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(plt)
    
    return None, corr

def main():
    configure_page()

    st.title("🌐 Network Traffic Analytics Dashboard")
    st.markdown(
        """
        📊 **Analyze CloudWatch Traffic:**
        - Explore feature correlations
        - Visualize detection patterns by country
        """
    )

    # Your existing code...

    tabs = st.tabs(
        ["📈 Correlation Heatmap", "🌎 Detection Types", "📥 Download Matrix"]
    )

    with tabs[0]:
        st.subheader("Correlation Heatmap (Filtered)")
        create_correlation_heatmap(features_df)  # Call the updated function

    with tabs[1]:
        st.subheader("Detection Frequency by Country (Filtered)")
        bar_fig = create_country_detection_barplot(filtered_df)
        st.plotly_chart(bar_fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Download Correlation Matrix")
        csv = convert_df_to_csv(corr_matrix)
        st.download_button(
            label="Download Correlation Matrix as CSV",
            data=csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()