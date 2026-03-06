import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


def load_data(path="UniversalBank.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=['ID', 'ZIP Code'], errors='ignore')
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Income_Group'] = pd.cut(df['Income'], bins=[0, 50, 100, 150, 200, 300],
                                 labels=['<50K', '50-100K', '100-150K', '150-200K', '200K+'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100],
                              labels=['<30', '30-40', '40-50', '50-60', '60+'])
    df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced'})
    df['Family_Label'] = df['Family'].map({1: '1', 2: '2', 3: '3', 4: '4+'})
    return df


def prepare_features(df):
    feature_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                    'Education', 'Mortgage', 'Securities Account',
                    'CD Account', 'Online', 'CreditCard']
    X = df[feature_cols]
    y = df['Personal Loan']
    return X, y, feature_cols


def train_models(df):
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=7)
    }

    results = {}
    for name, model in models.items():
        if name in ['Logistic Regression', 'KNN']:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name in ['Logistic Regression', 'KNN']:
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'confusion': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importance'] = pd.Series(
                model.feature_importances_, index=feature_cols
            ).sort_values(ascending=False)

    best_model_name = max(results, key=lambda x: results[x]['auc'])

    return results, best_model_name, X_test, y_test, scaler, feature_cols


def get_prescriptive_insights(df):
    """Generate prescriptive insights based on the data patterns."""
    insights = []

    # Income threshold
    accepted = df[df['Personal Loan'] == 1]
    rejected = df[df['Personal Loan'] == 0]

    inc_threshold = accepted['Income'].quantile(0.25)
    insights.append({
        'category': 'Income Targeting',
        'insight': f'Focus on customers earning above ${inc_threshold:.0f}K/year',
        'detail': f'75% of loan acceptors earn above ${inc_threshold:.0f}K. Average income of acceptors: ${accepted["Income"].mean():.0f}K vs ${rejected["Income"].mean():.0f}K for rejectors.',
        'action': 'Prioritize high-income segments in campaign outreach',
        'impact': 'High'
    })

    # CD Account
    cd_acceptance = df.groupby('CD Account')['Personal Loan'].mean()
    insights.append({
        'category': 'CD Account Holders',
        'insight': f'CD Account holders are {cd_acceptance[1]/cd_acceptance[0]:.1f}x more likely to accept',
        'detail': f'Acceptance rate: CD holders {cd_acceptance[1]*100:.1f}% vs non-holders {cd_acceptance[0]*100:.1f}%',
        'action': 'Cross-sell personal loans to existing CD account customers first',
        'impact': 'Very High'
    })

    # Education
    edu_acceptance = df.groupby('Education')['Personal Loan'].mean()
    insights.append({
        'category': 'Education Targeting',
        'insight': 'Graduate and Advanced degree holders show higher acceptance rates',
        'detail': f'Undergrad: {edu_acceptance[1]*100:.1f}% | Graduate: {edu_acceptance[2]*100:.1f}% | Advanced: {edu_acceptance[3]*100:.1f}%',
        'action': 'Tailor loan product messaging for educated professionals',
        'impact': 'Medium'
    })

    # Family size
    fam_acceptance = df.groupby('Family')['Personal Loan'].mean()
    best_fam = fam_acceptance.idxmax()
    insights.append({
        'category': 'Family Size',
        'insight': f'Families of size {best_fam} show the highest acceptance rate',
        'detail': f'Acceptance rate by family size: {dict(fam_acceptance.apply(lambda x: f"{x*100:.1f}%"))}',
        'action': 'Segment campaigns by family size — larger families may need loan products more',
        'impact': 'Medium'
    })

    # CCAvg
    cc_threshold = accepted['CCAvg'].quantile(0.25)
    insights.append({
        'category': 'Credit Card Spending',
        'insight': f'Higher credit card spenders (>${cc_threshold:.1f}K/mo) are better prospects',
        'detail': f'Average CC spending: Acceptors ${accepted["CCAvg"].mean():.2f}K vs Rejectors ${rejected["CCAvg"].mean():.2f}K/month',
        'action': 'Use credit card spending behavior as a filter for campaign targeting',
        'impact': 'High'
    })

    return insights
