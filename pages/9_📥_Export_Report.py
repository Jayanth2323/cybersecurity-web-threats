# from functools import reduce
# from operator import getitem
import streamlit as st
import pandas as pd
import datetime
from io import BytesIO

# import xlsxwriter as xw

# Page Configuration
st.set_page_config(
    page_title="📥 Export CSV/Excel Threat Report",
    layout="wide",
    menu_items={
        "Get Help": "https://example.com/help",
        "Report a bug": "https://example.com/bug",
        "About": "# Threat Report Generator",
    },
)
st.title("Generate & Download CSV/Excel Threat Report")


@st.cache_data
def load_data():
    """Load and validate the threat data."""
    try:
        df = pd.read_csv("data/analyzed_output.csv")
        if "anomaly" not in df.columns:
            st.error("Missing required 'anomaly' column in dataset")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Load data with error handling
df = load_data()
if df is None:
    st.stop()

# Sidebar for filters and metadata
with st.sidebar:
    st.header("Report Configuration")

    # Metadata
    report_title = st.text_input(
        "Report Title", "Cybersecurity Threat Summary"
    )
    analyst_name = st.text_input("Prepared By", "Security Team")
    report_date = st.date_input("Report Date", datetime.date.today())
    department = st.selectbox(
        "Department", ["All", "IT", "Finance", "HR", "Operations"]
    )

    # Additional filters
    st.header("Data Filters")
    status_options = df["anomaly"].dropna().unique().tolist()
    selected_status = st.multiselect(
        "Filter by Anomaly Status", status_options, default=status_options
    )

    if "severity" in df.columns:
        severity_options = df["severity"].dropna().unique().tolist()
        selected_severity = st.multiselect(
            "Filter by Severity", severity_options, default=severity_options
        )

# Apply filters
filtered_df = df[df["anomaly"].isin(selected_status)].copy()
if "severity" in df.columns and "selected_severity" in locals():
    filtered_df = filtered_df[filtered_df["severity"].isin(selected_severity)]
if department != "All":
    filtered_df = filtered_df[filtered_df["department"] == department]

# Main content area
st.header("Report Preview")
st.info(f"Showing {len(filtered_df)} of {len(df)} total records")

# Enhanced data preview with tabs
tab1, tab2 = st.tabs(["Data Preview", "Summary Statistics"])
with tab1:
    st.dataframe(
        filtered_df.style.apply(
            lambda x: [
                "background: #FFCCCC" if x.anomaly == "suspicious" else ""
                for _ in x
            ],
            axis=1,
        ),
        use_container_width=True,
        height=400,
    )

with tab2:
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(filtered_df))
            st.metric(
                "Suspicious Activity",
                f"{len(filtered_df[filtered_df['anomaly'] == 'suspicious'])} "
                f"""({len(filtered_df[filtered_df['anomaly'] == 'suspicious'])
                /len(filtered_df):.1%})""",
            )
        with col2:
            if "severity" in filtered_df.columns:
                st.write("Severity Distribution:")
                st.bar_chart(filtered_df["severity"].value_counts())
            else:
                st.write("Anomaly Distribution:")
                st.bar_chart(filtered_df["anomaly"].value_counts())


# Excel Export with Enhanced Formatting
def generate_excel_report(df, title):
    """Generate styled Excel report with conditional formatting."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        _extracted_from_generate_excel_report_7(df, writer, title)
    output.seek(0)
    return output


# TODO Rename this here and in `generate_excel_report`
def _extracted_from_generate_excel_report_7(df, writer, title):
    # Main data sheet
    df.to_excel(writer, index=False, sheet_name="Threat Report")
    workbook = writer.book
    worksheet = writer.sheets["Threat Report"]

    # Add header formatting
    header_format = workbook.add_format(
        {
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#4472C4",
            "font_color": "white",
            "border": 1,
        }
    )

    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # Auto-adjust column widths
    for i, col in enumerate(df.columns):
        max_len = max((df[col].astype(str).map(len).max(), len(str(col)))) + 2
        worksheet.set_column(i, i, min(max_len, 50))

    # Conditional formatting for anomalies
    suspicious_format = workbook.add_format({"bg_color": "#FFC7CE"})
    worksheet.conditional_format(
        1,
        0,
        len(df),
        0,
        {
            "type": "text",
            "criteria": "containing",
            "value": "suspicious",
            "format": suspicious_format,
        },
    )

    # Add summary sheet
    summary_df = pd.DataFrame(
        {
            "Report Title": [title],
            "Generated On": [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ],
            "Prepared By": [analyst_name],
            "Total Records": [len(df)],
            "Suspicious Count": [len(df[df["anomaly"] == "suspicious"])],
        }
    )
    summary_df.to_excel(writer, index=False, sheet_name="Report Summary")

    # Freeze headers
    worksheet.freeze_panes(1, 0)


# Export Buttons
st.header("Export Options")
col1, col2 = st.columns(2)

with col1:
    # CSV Export
    st.download_button(
        label="Download CSV Report",
        data=filtered_df.to_csv(index=False),
        file_name=f"""{
            report_title.replace(' ', '_')}_{
                report_date.strftime('%Y%m%d')}.csv""",
        mime="text/csv",
        help="Download filtered data as CSV file",
    )

with col2:
    # Excel Export
    if st.button("Generate Excel Report"):
        try:
            excel_data = generate_excel_report(filtered_df, report_title)
            st.download_button(
                label="Download Excel Report",
                data=excel_data,
                file_name=f"""{
                    report_title.replace(' ', '_')}_{
                        report_date.strftime('%Y%m%d')}.xlsx""",
                mime="""application/
                vnd.openxmlformats-officedocument.spreadsheetml.sheet""",
                help="""Download styled Excel report
                with conditional formatting""",
            )
        except Exception as e:
            st.error(f"Error generating Excel report: {str(e)}")

# Add footer
st.markdown("---")
st.caption(
    f"""
    Report generated on {
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
)
