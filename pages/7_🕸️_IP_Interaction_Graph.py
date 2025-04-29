# pages/7_🕸️_IP_Interaction_Graph.py
import contextlib
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional

# Configuration
DEFAULT_DATA_PATH = "data/analyzed_output.csv"


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="IP Interaction Network",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_data(file_path: str = DEFAULT_DATA_PATH) -> Optional[pd.DataFrame]:
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(file_path)
        required_cols = {"src_ip", "dst_ip", "anomaly"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.warning(f"Missing required columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None


def create_interaction_graph(
    df: pd.DataFrame,
) -> tuple[plt.Figure, nx.Graph]:
    """Generate interactive IP network visualization."""
    st.markdown("### Network Traffic Flow Visualization")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_only_suspicious = st.checkbox(
            "Show only Suspicious connections", False
        )
    with col2:
        max_nodes = st.slider("Maximum nodes to display", 10, 200, 50)

    # Filter data if needed
    if show_only_suspicious:
        df = df[df["anomaly"].str.lower() == "suspicious"]

    src_ips = df["src_ip"].dropna().unique()[:max_nodes]

    # Build graph with a central hub
    G = nx.Graph()
    central_node = "SRC_IPs"
    G.add_node(central_node)
    for ip in src_ips:
        G.add_edge(central_node, ip)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=1.5, ax=ax)

    # Draw labels with optional filtering for readability
    label_dict = {node: node for node in G.nodes() if node != central_node}
    nx.draw_networkx_labels(
        G, pos, labels=label_dict, font_size=8, font_color="navy", ax=ax
        )

    ax.set_title("Network Interaction between Source IPs", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    return fig, G


def show_graph_statistics(G: nx.Graph):
    """Display basic network statistics in sidebar."""
    st.sidebar.markdown("### Network Statistics")

    if not G or len(G.nodes) == 0:
        st.sidebar.warning("No nodes available in the graph.")
        return

    st.sidebar.metric("Total Nodes", len(G.nodes()))
    st.sidebar.metric("Total Connections", len(G.edges()))

    try:
        density = nx.density(G)
        st.sidebar.metric("Network Density", f"{density:.3f}")
    except nx.NetworkXError as e:
        st.sidebar.error(f"Failed to calculate network density: {str(e)}")

    with contextlib.suppress(Exception):
        if nx.is_weakly_connected(G):
            diameter = nx.diameter(G)
            st.sidebar.metric("Network Diameter", diameter)


def main():
    configure_page()
    st.title("🕸️ IP Interaction Network")
    st.markdown("Visualize communication patterns between network hosts")

    df = load_data()
    if df is None:
        return

    # Display unique src_ip's in sidebar
    st.sidebar.markdown("### Unique Source IPs")
    unique_src_ips = df["src_ip"].dropna().unique()
    st.sidebar.write(f"Total Unique Source IPs: {len(unique_src_ips)}")
    st.sidebar.dataframe(pd.DataFrame(unique_src_ips, columns=["src_ip"]))

    fig, graph = create_interaction_graph(df)
    st.pyplot(fig)
    show_graph_statistics(graph)

    # Add expander with raw data preview
    with st.expander("Show raw data preview"):
        st.dataframe(df[["src_ip", "dst_ip", "anomaly"]].head(100))


if __name__ == "__main__":
    main()
