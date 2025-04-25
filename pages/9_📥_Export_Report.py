import streamlit as st
import pandas as pd
import datetime
from io import BytesIO
# import xlsxwriter

st.set_page_config(page_title="📥 Export CSV/Excel Report", layout="wide")
st.title("Generate & Download CSV/Excel Threat Report")


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


# Function to style the Excel export with conditional formatting
def style_excel(df):
    # Create a BytesIO object to save the Excel file
    output = BytesIO()

    # Create an Excel writer with xlsxwriter engine
    writer = pd.ExcelWriter(output, engine="xlsxwriter")

    # Write the dataframe to the Excel file
    df.to_excel(writer, index=False, sheet_name="Threat Report")

    # Get the xlsxwriter workbook and sheet objects
    workbook = writer.book
    worksheet = writer.sheets["Threat Report"]

    # Define cell formatting for 'Suspicious' and 'Normal'
    suspicious_format = workbook.add_format(
        {"bg_color": "#FFCCCC"}
    )  # Red background for suspicious
    normal_format = workbook.add_format(
        {"bg_color": "#E6FFE6"}
    )  # Green background for normal

    # Iterate over the rows to apply conditional formatting
    for row_num, row in df.iterrows():
        if str(row["anomaly"]).lower() == "suspicious":
            worksheet.set_row(
                row_num + 1, None, suspicious_format
            )  # +1 because Excel is 1-indexed
        else:
            worksheet.set_row(row_num + 1, None, normal_format)

    # Save the Excel file
    writer.save()
    output.seek(0)
    return output


# CSV Export
if st.button("Generate CSV"):
    try:
        # Save filtered data to CSV
        csv_output = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name=f"""{report_title.replace(' ', '_')}_{
                datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv""",
            mime="text/csv",
        )
        st.success("✅ CSV generated and ready for download.")
    except Exception as e:
        st.error("⚠️ Unexpected error during CSV generation.")
        st.code(str(e))

# Excel Export
# if st.button("Generate Excel"):
#     try:
#         # Save filtered data to Excel with styles
#         excel_output = style_excel(filtered_df)
#         st.download_button(
#             label="Download Excel",
#             data=excel_output,
#             file_name=f"""
#             {report_title.replace(' ', '_')}_{
#                 datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx""",
#             mime="""application
#             /vnd.openxmlformats-officedocument.spreadsheetml.sheet""",
#         )
#         st.success("✅ Excel generated and ready for download.")
#     except Exception as e:
#         st.error("⚠️ Unexpected error during Excel generation.")
#         st.code(str(e))

# PDF Export
# st.set_page_config(page_title=" Export CSV/Excel Report", layout="wide")
# st.title("Generate & Download CSV/Excel Threat Report")


# @st.cache_data
# def load_data():
#     try:
#         return pd.read_csv("data/analyzed_output.csv")
#     except Exception as e:
#         st.error("Error loading data")
#         st.text(str(e))


# df = load_data()

# Filter Options
# # # st.markdown("### Filter Options")
# # # status_options = df["anomaly"].dropna().unique().tolist()
# # # selected_status = st.multiselect(
# # #     "Filter by Anomaly", status_options, default=status_options
# # # )
# # # filtered_df = df[df["anomaly"].isin(selected_status)].copy()

# # Metadata
# st.markdown("### Report Metadata")
# with st.form("report_metadata"):
#     report_title = st.text_input(
#         "Report Title", "Cybersecurity Threat Summary"
#     )
#     analyst_name = st.text_input("Prepared By", "Jayanth Chennoju")
#     report_date = st.date_input("Report Date", datetime.date.today())
#     submit_button = st.form_submit_button(label="Submit")

# Preview Data
st.markdown("### Preview of Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Function to style the Excel export with conditional formatting


# def style_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine="xlsxwriter")
#     df.to_excel(writer, index=False, sheet_name="Threat Report")
#     workbook = writer.book
#     worksheet = writer.sheets["Threat Report"]
#     suspicious_format = workbook.add_format({"bg_color": "#FFCCCC"})
#     normal_format = workbook.add_format({"bg_color": "#E6FFE6"})
#     for row_num, row in df.iterrows():
#         if str(row["anomaly"]).lower() == "suspicious":
#             worksheet.set_row(row_num + 1, None, suspicious_format)
#         else:
#             worksheet.set_row(row_num + 1, None, normal_format)
#     writer.save()
#     output.seek(0)
#     return output


# # CSV Export


# if st.button("Generate CSV"):
#     try:
#         csv_output = filtered_df.to_csv(index=False)
#         st.download_button(
#             label="Download CSV",
#             data=csv_output,
#             file_name=f"""{report_title.replace(' ', '_')}_{
#                 datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv""",
#             mime="text/csv",
#         )
#         st.success(" CSV generated and ready for download.")
#     except Exception as e:
#         st.error(" Unexpected error during CSV generation.")
#         st.code(str(e))
