# SHAP Force Plot
st.markdown("### SHAP Force Plot Explanation")
shap.initjs()

sample_shap = shap_values[selected_index]  # ✅ Use the already-built Explanation

force_plot = shap.force_plot(
    sample_shap.base_values,
    sample_shap.values,
    sample_shap.data,
    feature_names=sample_shap.feature_names,
    matplotlib=False,
)

shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
components.html(shap_html, height=300)
