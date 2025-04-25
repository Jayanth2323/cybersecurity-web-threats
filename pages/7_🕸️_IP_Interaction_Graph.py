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
) -> tuple[plt.Figure, nx.DiGraph]:
    """Generate interactive IP network visualization."""
    st.markdown("### Network Traffic Flow Visualization")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_only_suspicious = st.checkbox(
            "Show only suspicious connections", False
        )
    with col2:
        max_nodes = st.slider("Maximum nodes to display", 10, 200, 50)

    # Filter data if needed
    if show_only_suspicious:
        df = df[df["anomaly"] == "suspicious"]

    # Sample data if too large
    if len(df) > 1000:
        df = df.sample(min(1000, len(df)))

    # Create graph
    G = nx.from_pandas_edgelist(
        df,
        source="src_ip",
        target="dst_ip",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    # Limit graph size for performance
    if len(G.nodes()) > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[
            :max_nodes
        ]
        G = G.subgraph([n[0] for n in top_nodes])

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Node coloring based on degree centrality
    node_degrees = dict(G.degree())
    node_sizes = [v * 20 for v in node_degrees.values()]

    # Edge coloring based on anomaly status
    edge_colors = []
    for u, v, d in G.edges(data=True):
        if "anomaly" in d and d["anomaly"] == "suspicious":
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9, ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=1.0,
        arrows=True,
        arrowsize=10,
        ax=ax,
    )

    # Label only high-degree nodes
    high_degree_nodes = [n for n, d in node_degrees.items() if d > 3]
    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in high_degree_nodes}, font_size=8, ax=ax
    )

    ax.set_title(
        "IP Interaction Network (Red = Suspicious Connections)", fontsize=14
    )
    ax.axis("off")
    plt.tight_layout()

    return fig, G


def show_graph_statistics(G: nx.DiGraph):
    """Display network metrics."""
    st.sidebar.markdown("### Network Statistics")

    if len(G.nodes()) == 0:
        st.sidebar.warning("No nodes in the graph")
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

    fig, graph = create_interaction_graph(df)
    st.pyplot(fig)
    show_graph_statistics(graph)

    # fig = create_interaction_graph(df)
    # st.pyplot(fig)
    # show_graph_statistics(fig.graph)

    # Add expander with raw data preview
    with st.expander("Show raw data preview"):
        st.dataframe(df[["src_ip", "dst_ip", "anomaly"]].head(100))


if __name__ == "__main__":
    main()
