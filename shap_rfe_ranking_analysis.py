import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import shap

# 1. Data Loading & Preprocessing
try:
    df = pd.read_csv('final_dataset.csv')
except FileNotFoundError:
    print("Error: 'final_dataset.csv' not found.")
    exit()

X = df.drop(columns=['name', 'target']) if 'name' in df.columns else df.drop(columns=['target'])
y = df['target']

# Handle missing values
X = X.fillna(X.median())

# Format feature names to clean title case for presentation
cleaned_feature_names = [col.replace('_', ' ').title() for col in X.columns]
X.columns = cleaned_feature_names

# 2. Train Base Estimator
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 3. Recursive Feature Elimination (RFE) Ranking
rfe = RFE(estimator=rf, n_features_to_select=1, step=1)
rfe.fit(X, y)

rfe_df = pd.DataFrame({'Feature': X.columns, 'RFE_Ranking': rfe.ranking_})
rfe_df = rfe_df.sort_values(by='RFE_Ranking', ascending=True).reset_index(drop=True)

# 4. SHAP Values Calculation
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Handle different SHAP output format conventions
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

mean_shap_importance = np.abs(shap_vals_class1).mean(axis=0)
shap_df = pd.DataFrame({'Feature': X.columns, 'SHAP_Importance': mean_shap_importance})
shap_df = shap_df.sort_values(by='SHAP_Importance', ascending=False).reset_index(drop=True)

# 5. Terminal Reporting
print("=== FEATURE RANKING BY RFE (Rank 1 is the most important) ===")
for idx, row in rfe_df.iterrows():
    print(f"{idx+1}. {row['Feature']} (Rank: {row['RFE_Ranking']})")

print("\n=== FEATURE IMPORTANCE BY MEAN |SHAP| ===")
for idx, row in shap_df.iterrows():
    print(f"{idx+1}. {row['Feature']} (Mean |SHAP|: {row['SHAP_Importance']:.4f})")

# 6. Combined Performance Metric Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Left Plot: Mean SHAP Values
shap_df_sorted = shap_df.sort_values(by='SHAP_Importance', ascending=True)
axes[0].barh(shap_df_sorted['Feature'], shap_df_sorted['SHAP_Importance'], color='#1f77b4')
axes[0].set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=11)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# Right Plot: RFE Ranking Order
rfe_df_sorted = rfe_df.sort_values(by='RFE_Ranking', ascending=False)
max_rank = rfe_df_sorted['RFE_Ranking'].max()
axes[1].barh(rfe_df_sorted['Feature'], max_rank - rfe_df_sorted['RFE_Ranking'] + 1, color='#2ca02c')
axes[1].set_xlabel('RFE Score (Rank 1 is the Longest Bar)', fontsize=11)
axes[1].set_xticklabels([])  
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('combined_shap_rfe_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Standard SHAP Summary Plot Publication
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_class1, X, show=False) 
plt.tight_layout()
plt.savefig('standard_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSuccess: Graphs saved as 'combined_shap_rfe_analysis.png' and 'standard_shap_summary_plot.png'.")