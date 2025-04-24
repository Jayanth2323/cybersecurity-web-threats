# SHAP Force Plot (Corrected)
st.markdown("### 🔬 SHAP Force Plot Explanation")
shap.initjs()

# Corrected call to force plot
force_plot = shap.plots.force(
    explainer.expected_value, shap_values[selected_index].values
)

# Use the components library to display the force plot
shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
components.html(shap_html, height=300)
