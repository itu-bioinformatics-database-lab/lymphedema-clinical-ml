import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import os

# Set professional plotting style
sns.set_theme(style="whitegrid", palette="muted")

# --- 1. Load Dataset ---
try:
    df = pd.read_csv('final_dataset.csv')
except FileNotFoundError:
    print("Error: 'final_dataset.csv' not found. Please ensure it is in the same directory.")
    exit()

# --- 2. Fit Logistic Regression Models ---
def fit_logistic_model(data, intervention_type):
    # Filter and drop missing values for data consistency
    sub = data[data['axillary_intervention'] == intervention_type].dropna(subset=['bmi', 'target'])
    
    if len(sub) < 10: 
        return None, None
        
    X = sub[['bmi']]
    y = sub['target']
    
    # Fit Logistic Regression model
    model = LogisticRegression().fit(X, y)
    
    # Generate a smooth array of 100 points for BMI ranging from 18 to 48
    bmi_range = pd.DataFrame(np.linspace(18, 48, 100), columns=['bmi'])
    
    # Predict probabilities (column 1 represents the probability of the target event)
    probs = model.predict_proba(bmi_range)[:, 1]
    
    return bmi_range.values.flatten(), probs

# --- 3. Model Execution ---
bmi_x, prob_slnb = fit_logistic_model(df, 0)
_, prob_alnd = fit_logistic_model(df, 1)

# --- 4. Professional Visualization ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot SLNB (Low Burden Curve)
if prob_slnb is not None:
    ax.plot(bmi_x, prob_slnb, color='#2980B9', linewidth=3, label='SLNB (Low Burden)')
    ax.fill_between(bmi_x, prob_slnb, alpha=0.15, color='#2980B9')

# Plot ALND (High Burden Curve)
if prob_alnd is not None:
    ax.plot(bmi_x, prob_alnd, color='#C0392B', linewidth=3, label='ALND (High Burden)')
    ax.fill_between(bmi_x, prob_alnd, alpha=0.15, color='#C0392B')

# Axes Label Configurations
ax.set_xlabel("Body Mass Index (BMI)", fontsize=12, labelpad=10)
ax.set_ylabel("Estimated Risk Probability", fontsize=12, labelpad=10)

# Adjust legend coordinates outside to the right center
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=11, borderpad=1)

# Clean up axes and borders for a modern look
sns.despine(left=True, bottom=True)
ax.tick_params(axis='both', which='major', labelsize=11)

# Save high-quality visualization (300 DPI)
save_path = "bmi_surgical_threshold.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Plot successfully saved to: {save_path}\n")

# --- 5. Textual Analysis of the Divergence Trend ---
if prob_slnb is not None and prob_alnd is not None:
    risk_diff = prob_alnd - prob_slnb
    
    print("--- DYNAMIC RISK DIFFERENCE ANALYSIS ---")
    print("Risk probability gap (ALND - SLNB) relative to BMI progression:")
    print("-" * 50)
    
    for target_bmi in [18, 25, 30, 35, 40, 48]:
        idx = np.abs(bmi_x - target_bmi).argmin()
        diff = risk_diff[idx] * 100
        alnd_risk = prob_alnd[idx] * 100
        slnb_risk = prob_slnb[idx] * 100
        print(f"At BMI {target_bmi:02d} | SLNB Risk: {slnb_risk:05.2f}% | ALND Risk: {alnd_risk:05.2f}% | Absolute Difference: {diff:.2f}%")
        
    print("-" * 50)
    
    # Evaluate the structural divergence trend of both curves
    if risk_diff[-1] > risk_diff[0]:
        print("Trend Conclusion: The risk discrepancy between the two surgical approaches WIDENS as BMI increases.")
    else:
        print("Trend Conclusion: The risk discrepancy between the two surgical approaches NARROWS as BMI increases.")