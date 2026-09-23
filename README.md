# Mental Wellbeing Check

A machine learning web application that estimates a person's **Health & Wellbeing Score** based on daily lifestyle and behavioural habits.

> **Note:** This application is designed for educational purposes and is not a medical diagnostic tool. It does not diagnose depression, anxiety, or any medical condition.

---

## Features

- **Wellbeing Assessment** — Answer questions about your daily habits (sleep, activity, nutrition, social life, work-life balance)
- **ML-Powered Scoring** — A trained regression model predicts your overall wellbeing score (0–100)
- **Feature Engineering** — Five composite dimension scores: Sleep Quality, Physical Activity, Nutrition, Social Connection, and Daily Balance
- **Explainable Results** — See exactly which factors are affecting your score the most and get personalised recommendations
- **Assessment History** — All past assessments are stored locally using SQLite

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML | Scikit-learn, XGBoost, SHAP |
| Data | Pandas, NumPy |
| Database | SQLite |
| Frontend | HTML, CSS, Bootstrap 5 |
| Visualisation | Matplotlib, Seaborn |

---

## Project Structure

```
Mental_Wellbeing_Check/
├── app.py                    # Flask application and routes
├── config.py                 # Central configuration
├── database.py               # SQLite operations
├── feature_engineering.py    # Composite score calculations
├── train_model.py            # Model training pipeline
├── model.py                  # Model loading interface
├── predict.py                # Prediction + SHAP + recommendations
├── eda.py                    # Exploratory data analysis
├── utils.py                  # Helper functions
├── requirements.txt
├── README.md
├── data/
│   └── mental_wellbeing_dataset.csv
├── model/
│   ├── wellbeing_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── training_report.json
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── assess.html
│   ├── result.html
│   ├── history.html
│   └── about.html
└── static/
    ├── css/style.css
    ├── js/main.js
    └── eda/
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Mental-Wellbeing-Check.git
cd Mental-Wellbeing-Check
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Engineer composite features
- Train Linear Regression, Random Forest, and XGBoost
- Compare models using MAE, RMSE, and R² Score
- Save the best model to `model/`

### 5. (Optional) Run EDA

```bash
python eda.py
```

Generates exploratory data analysis plots in `static/eda/`.

### 6. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Machine Learning Pipeline

1. **Preprocessing** — Handle missing values (median for numerical, mode for categorical)
2. **Feature Engineering** — Create five composite wellbeing scores
3. **Encoding** — Convert categorical variables using fixed ordinal mapping
4. **Scaling** — StandardScaler applied to all features
5. **Training** — Three models trained and compared:
   - Linear Regression
   - Random Forest Regressor (200 trees, max depth 12)
   - XGBoost Regressor (200 trees, max depth 6, lr 0.1)
6. **Evaluation** — MAE, RMSE, R² Score on 20% held-out test set:

   | Model | MAE | RMSE | R² |
   |-------|-----|------|----|
   | **Linear Regression** | 7.71 | 9.78 | **0.387** ✅ Winner |
   | XGBoost | 7.77 | 9.87 | 0.376 |
   | Random Forest | 8.03 | 10.13 | 0.342 |

7. **Final Model** — **Linear Regression** achieved the highest R² (0.387) and is saved as `model/wellbeing_model.pkl`

---

## Model Explainability

Each prediction includes **feature importance explanations** generated using SHAP (SHapley Additive exPlanations). The top 5 most impactful features are shown with:
- Human-readable descriptions
- Positive/negative impact indicators
- Personalised actionable recommendations

---

## Screenshots

*Run the application to see the UI.*

---

## License

This project is for educational purposes only.
