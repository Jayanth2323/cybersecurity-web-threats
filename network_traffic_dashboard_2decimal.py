def create_correlation_heatmap(
    df: pd.DataFrame,
) -> Tuple[px.imshow, pd.DataFrame]:
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",  # Change format to display 2 decimal places
        cmap="coolwarm",
        annot_kws={"size": 12},  # Increase font size here
        cbar_kws={"shrink": 0.82},
    )

    # Show the plot in Streamlit
    st.pyplot(plt)

    fig = px.imshow(
        corr,
        text_auto=lambda z: f'{z:.2f}',  # Format correlation values to 2 decimal places
        color_continuous_scale=[
            (1.0, "maroon"),
            (0.5, "lightblue"),
            (0.0, "blue"),
            (0.3, "lightblue"),  # Added for a smoother gradient
            (0.2, "lightblue"),
            (0.1, "lightblue"),
            (0.0, "blue"),
        ],
        title="Correlation Matrix Heatmap",
        aspect="auto",
        height=900,
        width=1200,
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticks="outside",
        ),
        title=dict(font=dict(size=24)),
        xaxis_title=dict(font=dict(size=18)),
        yaxis_title=dict(font=dict(size=18)),
    )

    fig.update_traces(textfont=dict(size=10))

    return fig, corr