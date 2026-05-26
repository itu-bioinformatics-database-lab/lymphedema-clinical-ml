import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 0. Warnings Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, 
                              BaggingClassifier, StackingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, 
                             roc_auc_score, make_scorer, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# --- Custom Metric Definitions ---
def npv_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0

scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision', # PPV
    'recall': 'recall',       # Sensitivity
    'auc': 'roc_auc',
    'f1': 'f1',
    'npv': make_scorer(npv_score)
}

# 1. Setup Environment
folder_name = "ML_Analysis_Results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

plt.rcParams.update({'figure.dpi': 300, 'font.size': 10})

# 2. Data Loading
df = pd.read_csv("final_dataset.csv")
X = df.drop(columns=['name', 'target'])
y = df['target']

# Feature Groups for Combinations
groups = {
    "Demographics": ["age", "bmi", "tracking_period", "obesity_over_30"],
    "Tumor_Profile": ["biomarker", "er_percent", "pr_percent", "ki67_percent", "histological_grade", "tumor_stage"],
    "Nodal_Status": ["lymph_node_positive", "nodal_stage", "positive_lymph_nodes", "removed_lymph_nodes", "capsular_invasion", "metastasis_type"],
    "Treatment": ["treatment_setting", "surgery_type", "chemotherapy", "axillary_radiotherapy", "axillary_intervention"],
    "LDex_Metrics": ["ldex", "ldex7", "ldex10"]
}

# 3. Model Definitions
def get_base_models():
    return {
        "RandomForest": (RandomForestClassifier(n_jobs=-1), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
        "LogisticRegression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
        "SVM": (SVC(probability=True), {'C': [0.1, 1], 'kernel': ['rbf']}),
        "XGBoost": (XGBClassifier(n_jobs=-1, verbosity=0), {'learning_rate': [0.01, 0.1], 'n_estimators': [100]}),
        "AdaBoost": (AdaBoostClassifier(), {'n_estimators': [50, 100]}),
        "GBM": (GradientBoostingClassifier(), {'learning_rate': [0.05, 0.1], 'n_estimators': [100]}),
        "BaggedLR": (BaggingClassifier(estimator=LogisticRegression(), n_jobs=-1), {'n_estimators': [10, 50]}),
        "StochasticGB": (GradientBoostingClassifier(subsample=0.8), {'n_estimators': [100]}),
        "RF_Boosting": (XGBClassifier(booster='gbtree', tree_method='hist', n_jobs=-1, verbosity=0), {'max_depth': [6]})
    }

# 4. Redundancy Analysis
def analyze_redundancy(X, folder):
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Redundancy Analysis")
    plt.savefig(f"{folder}/feature_redundancy.png", dpi=300)

# 5. ML Engine
results = []
# Increased n_splits to 10 for more robust evaluation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

group_names = list(groups.keys())
all_combos = []
for r in range(1, len(group_names) + 1):
    all_combos.extend(combinations(group_names, r))

print("Analysis starting with updated groups and metrics...")
analyze_redundancy(X, folder_name)

for combo in all_combos:
    combo_name = "+".join(combo)
    current_features = [col for g in combo for col in groups[g]]
    X_subset = X[current_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    base_models = get_base_models()
    trained_estimators = []
    
    for name, (model, params) in base_models.items():
        # Hyperparameter optimization via GridSearchCV (cv=5 to balance runtime and performance)
        grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_scaled, y)
        best_model = grid.best_estimator()
        
        cv_res = cross_validate(best_model, X_scaled, y, cv=skf, scoring=scoring_metrics, n_jobs=-1)
        
        results.append({
            'Group': combo_name, 'Model': name,
            'Accuracy': cv_res['test_accuracy'].mean(),
            'PPV (Precision)': cv_res['test_precision'].mean(),
            'Recall (Sensitivity)': cv_res['test_recall'].mean(),
            'NPV': cv_res['test_npv'].mean(),
            'AUC': cv_res['test_auc'].mean(),
            'F1': cv_res['test_f1'].mean()
        })
        trained_estimators.append((name, best_model))

    # Meta-Learning (Stacking Approach)
    meta_models = {"Stack_RidgeLR": RidgeClassifierCV(), "Stack_XGB": XGBClassifier(n_jobs=-1, verbosity=0)}
    
    for m_name, m_model in meta_models.items():
        # Increased internal cross-validation folds to 10 for the StackingClassifier
        stack = StackingClassifier(estimators=trained_estimators, final_estimator=m_model, cv=10, n_jobs=-1)
        cv_res_stack = cross_validate(stack, X_scaled, y, cv=skf, scoring=scoring_metrics, n_jobs=-1)
        results.append({
            'Group': combo_name, 'Model': m_name,
            'Accuracy': cv_res_stack['test_accuracy'].mean(),
            'PPV (Precision)': cv_res_stack['test_precision'].mean(),
            'Recall (Sensitivity)': cv_res_stack['test_recall'].mean(),
            'NPV': cv_res_stack['test_npv'].mean(),
            'AUC': cv_res_stack['test_auc'].mean(),
            'F1': cv_res_stack['test_f1'].mean()
        })

# 6. Save Results & Visualization
results_df = pd.DataFrame(results)
results_df.to_csv(f"{folder_name}/all_model_metrics.csv", index=False)

plot_metrics = ['AUC', 'Accuracy', 'PPV (Precision)', 'Recall (Sensitivity)', 'NPV']
for metric in plot_metrics:
    plt.figure(figsize=(14, 8))
    top_models = results_df.sort_values(metric, ascending=False).head(10)
    
    ax = sns.barplot(data=top_models, x=metric, y='Model', hue='Group', palette='viridis')
    
    # Set x-axis ticks interval to 0.1
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    # Annotate bar values on the plot
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                    f'{width:.3f}', va='center', fontsize=9)

    plt.title(f"Top 10 Performers by {metric}")
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/top_performers_{metric.split()[0]}.png", dpi=300)

print(f"Done! Results saved in: {folder_name}")