
# 🛡️ Suspicious Web Threat Interaction Dashboard

An enterprise-grade, real-time dashboard for web threat detection, data filtering, ML explainability, and report generation — powered by Random Forests, Neural Networks, SHAP, and Streamlit.

---

## 🎯 Key Features
- 🔍 **Advanced Anomaly Detection** (Random Forests + Neural Networks)
- 🌍 **Geo-Mapping** of suspicious IP activity
- 📦 **Packet Size and Protocol Filters**
- 📈 **Interactive SHAP Explainability Dashboard**
- ⚖️ **Model Comparison** (Accuracy, F1 Score, ROC Curves)
- 📥 **PDF Report Generator** with custom filters and metadata
- 🧠 **Visual Analytics**, Correlation Heatmaps, Time-Series, and Port Analysis
- 🔐 **Optional User Login** for Secured Dashboards
- ✅ **CI/CD-Ready** for Streamlit Cloud Deployment with GitHub Actions

---

## 🚀 Live Demo
[![Streamlit App](https://img.shields.io/badge/Launch%20App-Streamlit-brightgreen)](https://cybersecurity-web-threats-cdjl6zk84ozvkcvbbw8wnj.streamlit.app/)

Experience the full capabilities of the dashboard live in action.

---

## 🧠 Technologies Used
- **Python 3.10**
- **Streamlit** for UI/UX and web deployment
- **pandas**, **scikit-learn**, **matplotlib**, **plotly**, **seaborn** for data manipulation and visualization
- **SHAP** for model explainability
- **pdfkit**, **xlsxwriter** for export functionalities
- **streamlit-authenticator** for user authentication
- **Docker** + **GitHub CI** (optional for containerized deployment)

---

## ⚙️ Setup Instructions

### 📦 Clone & Install
```bash
git clone https://github.com/your-username/cybersecurity-web-threats.git
cd cybersecurity-web-threats
pip install -r requirements.txt
```

### ▶️ Run Locally
```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
cybersecurity-web-threats/
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
```

---
## 👨‍💻 Author
**Jayanth Chennoju** — [Jayanth2323](https://github.com/Jayanth2323).

**LinkedIn** — [**🪪-Jayanth Chennoju**](https://linkedin.com/in/jayanth-chennoju-5a738923k/).
 
I warmly welcome contributions from the community. Please feel free to fork the project, make improvements, or submit a pull request.

Feel free to collaborate with us:
**Mail id** — .[**📧**](jayanthchennoju@gmail.com).
---

## 📄 License
This project is licensed under the **MIT License**.  
Please take a look at the [LICENSE](LICENSE) file for detailed information.

---
