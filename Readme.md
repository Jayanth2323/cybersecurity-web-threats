🛡️ Suspicious Web Threat Interaction Dashboard

An enterprise-grade dashboard for real-time web threat detection, data filtering, ML explainability, and report generation — powered by Random Forests, Neural Networks, SHAP, and Streamlit.

🎯 Key Features
🔍 Advanced anomaly detection (RF + NN)

🌍 Geo-mapping of suspicious IP activity

📦 Packet size and protocol filters

📈 Interactive SHAP explainability dashboard

⚖️ Model comparison (Accuracy, F1, ROC)

📥 PDF report generator with filters + metadata

🧠 Visual analytics, correlation, time-series, and port analysis

🔐 (Optional) User login for secured dashboards

✅ CI/CD-ready for Streamlit Cloud and GitHub Actions

🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Launch%20App-Streamlit-brightgreen)](https://cybersecurity-web-threats-cdjl6zk84ozvkcvbbw8wnj.streamlit.app/)

🧠 Technologies Used
Python 3.10

Streamlit

pandas, scikit-learn, matplotlib, plotly, seaborn

SHAP, pdfkit, xlsxwriter, streamlit-authenticator

Docker + GitHub CI (optional)

⚙️ Setup Instructions
📦 Clone & Install
bash
Copy
Edit
git clone https://github.com/your-username/cybersecurity-web-threats.git
cd cybersecurity-web-threats
pip install -r requirements.txt
▶️ Run Locally
bash
Copy
Edit
streamlit run streamlit_app.py

📦 cybersecurity-web-threats
├── streamlit_app.py
├── requirements.txt
├── pages/
│   ├── 1_📊_Visual_Analytics.py
│   ├── 2_📈_Model_Insights.py
│   ├── 3_🧠_Neural_Network.py
│   ├── 4_📦_Protocol_Port_Analysis.py
│   ├── 5_📊_Correlation_Heatmap.py
│   ├── 6_📈_Time_Series_Traffic.py
│   ├── 7_🕸️_IP_Interaction_Graph.py
│   ├── 8_🎯_Model_Comparison.py
│   ├── 9_📥_Export_Report.py
│   └── 10_🧠_SHAP_Model_Explainability.py
├── models/
│   ├── rf_model.py
│   └── nn_model.py
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model_eval.py
├── data/
│   └── analyzed_output.csv

👨‍💻 Author
Jayanth Chennoju — Jayanth2323

Feel free to fork, improve, or contribute!

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more info.