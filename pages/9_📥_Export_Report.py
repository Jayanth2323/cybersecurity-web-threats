import streamlit as st
import pandas as pd
import datetime
from xhtml2pdf import pisa
from io import BytesIO

st.set_page_config(page_title="📥 Export PDF Report", layout="wide")
st.title("Generate & Download PDF Threat Report")


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/analyzed_output.csv")
    except Exception as e:
        st.error("Error loading data")
        st.text(str(e))


df = load_data()

# Filter Options
st.markdown("### Filter Options")
status_options = df["anomaly"].dropna().unique().tolist()
selected_status = st.multiselect(
    "Filter by Anomaly", status_options, default=status_options
)
filtered_df = df[df["anomaly"].isin(selected_status)].copy()

# Metadata
st.markdown("### Report Metadata")
report_title = st.text_input("Report Title", "Cybersecurity Threat Summary")
analyst_name = st.text_input("Prepared By", "Jayanth Chennoju")
report_date = st.date_input("Report Date", datetime.date.today())

# Preview Data
st.markdown("### Preview of Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Build HTML
html = f"""
<html>
<head>
    <style>
        body {{
            font-family: 'Helvetica', sans-serif;
            padding: 30px;
            font-size: 12px;
        }}
        h1 {{
            color: #2E86C1;
            text-align: center;
            margin-bottom: 30px;
        }}
        .meta {{
            margin-bottom: 20px;
            font-size: 11pt;
        }}
        .meta strong {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #bbb;
            padding: 5px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f0f0f0;
        }}

        h1 {{ color: #ff4b4b; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; font-size: 10pt; }}
        th {{ background-color: #f4f4f4; }}
        body {{ font-family: Arial, sans-serif; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    <div class="meta">
        <p><strong>Analyst:</strong> {analyst_name}</p>
        <p><strong>Date:</strong> {report_date}</p>
        <p><strong>Total Records:</strong> {len(filtered_df)}</p>
    </div>
    {filtered_df.to_html(index=False, border=0)}

    <p><strong>Analyst:</strong> {analyst_name}<br>
    <strong>Date:</strong> {report_date}</p>
    <p><strong>Total Records:</strong> {len(filtered_df)}</p>
    {filtered_df.to_html(index=False, classes='table table-bordered')}
</body>
</html>
"""

# PDF Export with xhtml2pdf
if st.button("Generate PDF"):
    try:
        pdf_output = BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=pdf_output)
        if not pisa_status.err:
            st.download_button(
                label="Download PDF",
                data=pdf_output.getvalue(),
                file_name=f"""
                {report_title}_{
                    datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf""",
                mime="application/pdf",
            )
            st.success("PDF generated successfully!")
        else:
            st.error("""PDF generation failed.
                    Please check your data or formatting.""")

    except Exception as e:
        st.error("Unexpected error during PDF generation.")
        st.code(str(e))
