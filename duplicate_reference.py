# SHAP Force Plot (v0.20+ Compliant)
st.markdown("### SHAP Force Plot Explanation")
shap.initjs()

# Extract sample SHAP explanation
sample_shap = shap_values[selected_index]

# Build the SHAP Explanation object explicitly
explanation = shap.Explanation(
    values=sample_shap.values,
    base_values=sample_shap.base_values,
    data=sample.values[0],
    feature_names=sample.columns.tolist()
)

# Generate the force plot using v0.20+ signature
force_plot = shap.plots.force(
    explanation.base_values,
    explanation.values,
    explanation.data,
    feature_names=explanation.feature_names,
    matplotlib=False,
)

# Embed SHAP force plot in Streamlit via HTML
shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
components.html(shap_html, height=300)
