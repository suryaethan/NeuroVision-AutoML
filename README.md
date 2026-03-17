# 🧠 NeuroVision-AutoML

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-AutoML-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=for-the-badge&logo=streamlit)

> **The world's most autonomous AI/ML pipeline** — Drop any CSV dataset, and NeuroVision automatically cleans, engineers features, selects the best model, trains it, evaluates it, and explains every prediction using SHAP. Zero code required by the user.

---

## 🚀 What Makes This Unique?

| Feature | NeuroVision | Ordinary AutoML |
|---|---|---|
| Auto problem detection (classification/regression) | ✅ | ❌ |
| Smart feature engineering | ✅ | ❌ |
| SHAP explainability for every prediction | ✅ | ❌ |
| Real-time Streamlit dashboard | ✅ | ❌ |
| Model comparison leaderboard | ✅ | Partial |
| Confidence scoring per prediction | ✅ | ❌ |
| Anomaly detection built-in | ✅ | ❌ |

---

## 🏗️ Project Structure

```
NeuroVision-AutoML/
├── neurovision/
│   ├── __init__.py
│   ├── engine.py          # 🧠 Core AutoML orchestration engine
│   ├── preprocessor.py    # 🔧 Smart data cleaning & feature engineering
│   ├── trainer.py         # 🏋️ Multi-model training & comparison
│   ├── explainer.py       # 💡 SHAP-based AI explainability
│   └── anomaly.py         # 🔍 Isolation Forest anomaly detection
├── app.py                 # 🌐 Streamlit interactive dashboard
├── main.py                # 🚀 CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/suryaethan/NeuroVision-AutoML.git
cd NeuroVision-AutoML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run CLI on your dataset
python main.py --data your_data.csv --target your_target_column

# 4. Launch interactive dashboard
streamlit run app.py
```

---

## 🤖 Models Supported

- **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN, Gradient Boosting
- **Regression**: Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost Regressor, SVR
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor

---

## 📊 How It Works

```
CSV Input
   ↓
🔧 Smart Preprocessing (null handling, encoding, scaling)
   ↓
🧬 Feature Engineering (interaction terms, polynomial, datetime)
   ↓
🤖 Auto Problem Detection (classification vs regression)
   ↓
🏋️ Train All Models in Parallel
   ↓
🏆 Model Leaderboard (accuracy, F1, AUC, RMSE)
   ↓
💡 SHAP Explainability (global + per-prediction)
   ↓
📈 Streamlit Dashboard (interactive visualization)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Anomaly Detection | Isolation Forest |

---

## 📈 Example Output

```
========================================
🧠 NeuroVision AutoML Pipeline
========================================
✅ Dataset loaded: 1000 rows × 15 columns
🔧 Preprocessing complete
🤖 Problem Type Detected: CLASSIFICATION
🏋️ Training 7 models...

📊 MODEL LEADERBOARD:
  1. XGBoost          Accuracy: 96.4%  F1: 0.961  AUC: 0.989
  2. LightGBM         Accuracy: 95.8%  F1: 0.955  AUC: 0.985
  3. Random Forest    Accuracy: 94.2%  F1: 0.939  AUC: 0.978
  ...

🏆 Best Model: XGBoost
💡 Generating SHAP explanations...
✅ Dashboard ready at http://localhost:8501
========================================
```

---

## 👨‍💻 Author

**Surya Ethan** — ETL Engineer & AI/ML Developer
- GitHub: [@suryaethan](https://github.com/suryaethan)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

⭐ **Star this repo** if you find it useful!
