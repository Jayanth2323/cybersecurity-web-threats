import streamlit as st
import pandas as pd
import pdfkit
import datetime

# import os

st.set_page_config(page_title="📥 Export PDF Report", layout="wide")
st.title("📄 Generate & Download PDF Threat Report")


@st.cache_data
def load_data():
    return pd.read_csv("data/analyzed_output.csv")


df = load_data()

# Filter Options
st.markdown("### 🔍 Filter Options")
status_options = df["anomaly"].dropna().unique().tolist()
selected_status = st.multiselect(
    "Filter by Anomaly", status_options, default=status_options
)
filtered_df = df[df["anomaly"].isin(selected_status)].copy()

# Metadata
st.markdown("### 📝 Report Metadata")
report_title = st.text_input("Report Title", "Cybersecurity Threat Summary")
analyst_name = st.text_input("Prepared By", "Jayanth Chennoju")
report_date = st.date_input("Report Date", datetime.date.today())

# Preview Data
st.markdown("### 🧾 Preview of Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Build HTML
html = f"""
<html>
<head>
    <style>
        h1 {{ color: #ff4b4b; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; font-size: 10pt; }}
        th {{ background-color: #f4f4f4; }}
        body {{ font-family: Arial, sans-serif; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    <p><strong>Analyst:</strong> {analyst_name}<br>
    <strong>Date:</strong> {report_date}</p>
    <p><strong>Total Records:</strong> {len(filtered_df)}</p>
    {filtered_df.to_html(index=False, classes='table table-bordered')}
</body>
</html>
"""

# PDF Export
if st.button("📄 Generate PDF"):
    try:
        path_pdf = f"""{report_title}_{
            datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            }.pdf"""
        pdfkit.from_string(html, path_pdf, options={"quiet": ""})
        with open(path_pdf, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF",
                data=f,
                file_name=path_pdf,
                mime="application/pdf",
            )
            st.success("PDF generated successfully!")
    except Exception as e:
        st.error("❌ PDF generation failed. Is wkhtmltopdf installed?")
        st.text(str(e))
        st.text("Error Details:")
        st.code(str(e))
