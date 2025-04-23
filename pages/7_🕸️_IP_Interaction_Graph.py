# pages/7_\ud83d\udd78\ufe0f_IP_Interaction_Graph.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.title("\ud83d\udd78\ufe0f IP Interaction Network")


@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")


df = load_data()

if 'src_ip' in df.columns and 'dst_ip' in df.columns:
    st.markdown("Visualizing source-to-destination IP interactions.")

    # Build the network
    G = nx.from_pandas_edgelist(
        df, 'src_ip', 'dst_ip', create_using=nx.DiGraph())

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.3)
    nx.draw(
        G, pos, node_color='skyblue', node_size=500,
        with_labels=False, arrows=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Missing 'src_ip' or 'dst_ip' columns in the dataset.")
