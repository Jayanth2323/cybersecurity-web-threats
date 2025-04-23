# Sidebar - dynamic field filters
selected_values = {}

if 'dest_ip_country_code' in df.columns:
    fields = ["protocol", "dest_ip_country_code"]
    some_options = {
        field: df[field].dropna().unique().tolist()
        for field in fields
    }

    for i, field in enumerate(fields):
        selected_values[field] = st.sidebar.multiselect(
            f"Select {field} Options",
            options=some_options[field],
            default=some_options[field],
            key=f"multiselect_{field}_{i}",
        )
