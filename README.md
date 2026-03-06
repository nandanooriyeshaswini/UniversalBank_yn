# 🏦 UniversalBank Personal Loan Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing and predicting personal loan acceptance using the UniversalBank dataset.

## 📊 Dashboard Features

### Analysis Sections
| Tab | Analysis Type | Contents |
|-----|--------------|----------|
| 📊 Descriptive | Overview & Statistics | Acceptance rates, summary stats, distributions, correlation heatmap |
| 🔍 Diagnostic | Root Cause Analysis | Key drivers, banking services impact, income & education breakdowns |
| 🤖 Predictive | ML Model Performance | 5 classifiers compared, ROC curves, confusion matrix, feature importance |
| 💡 Prescriptive | Actionable Insights | Campaign strategies, targeting simulation, lift analysis |
| 🎯 Predict | Single Customer | Real-time prediction with probability gauge |

### Key Features
- **Interactive Filters**: Income range, education level, family size, loan status
- **Drill-down Visualization**: Donut chart with configurable breakdown (Education, Income Group, Family Size, Age Group)
- **5 Classification Models**: Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, KNN
- **Campaign Simulation**: Estimate targeting lift for different customer segments
- **Customer Predictor**: Input customer details and get real-time loan acceptance prediction

## 🚀 Deploy on Streamlit Cloud

### Step 1: Upload to GitHub
1. Create a new GitHub repository (e.g., `universalbank-loan-dashboard`)
2. Upload all files from this folder to the repository root:
   - `app.py`
   - `model_utils.py`
   - `requirements.txt`
   - `UniversalBank.csv`
   - `README.md`

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repository and set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

## 🗂️ File Structure
```
universalbank-loan-dashboard/
├── app.py                 # Main Streamlit dashboard
├── model_utils.py         # Data loading, model training, insights
├── requirements.txt       # Python dependencies
├── UniversalBank.csv      # Dataset
└── README.md              # This file
```

## 📈 Models Used
- **Random Forest** (200 estimators)
- **Gradient Boosting** (200 estimators)
- **Logistic Regression** (L2 regularization)
- **Decision Tree** (max depth 8)
- **K-Nearest Neighbors** (k=7)

Evaluation metrics: ROC-AUC, Accuracy, F1 Score, 5-fold Cross-Validation AUC

## 📋 Dataset Variables

| Variable | Description |
|----------|-------------|
| Age | Customer's age in completed years |
| Experience | Years of professional experience |
| Income | Annual income (K$) |
| Family | Family size |
| CCAvg | Avg monthly credit card spending (K$) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage value (K$) |
| Securities Account | Has securities account (0/1) |
| CD Account | Has certificate of deposit account (0/1) |
| Online | Uses internet banking (0/1) |
| CreditCard | Has UniversalBank credit card (0/1) |
| **Personal Loan** | **Target: Accepted loan offer (0/1)** |

## 🛠️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
