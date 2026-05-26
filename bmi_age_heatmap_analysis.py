import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset
try:
    df = pd.read_csv('final_dataset.csv')
    # Drop 'name' column if it exists in the dataframe
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Clean data: Convert to numeric and fill missing values with 0
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------------------------------------------------
# VARIABLES Configuration
# ---------------------------------------------------------
bmi_col = 'bmi'       
age_col = 'age'       
target_col = 'target'  # Target outcome (e.g., Lymphedema status: 0 or 1)

# 2. Data Preprocessing for Risk Matrix

# Define BMI bins and professional clinical labels
bmi_bins = [0, 25, 30, 60]
bmi_labels = ['Normal (<25)', 'Overweight (25-30)', 'Obese (>30)']
df['BMI_Group'] = pd.cut(df[bmi_col], bins=bmi_bins, labels=bmi_labels)

# Define Age bins and professional clinical labels
age_bins = [0, 45, 65, 120]  # Safe upper bound set to 120 instead of 100
age_labels = ['Young (<45)', 'Middle-Aged (45-65)', 'Older (>65)']
df['Age_Group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels)

# Calculate the mean of the target for each intersection to get the risk probability
# 'observed=False' is passed to suppress categorical grouping warnings in future pandas versions
risk_matrix = df.groupby(['BMI_Group', 'Age_Group'], observed=False)[target_col].mean().unstack()

# Convert risk probabilities to percentages
risk_matrix_pct = risk_matrix * 100

# ---------------------------------------------------------
# ANALYSIS 1: VISUAL RISK MATRIX (HEATMAP)
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(
    risk_matrix_pct, 
    annot=True, 
    cmap='Reds', 
    fmt='.1f', 
    cbar_kws={'label': 'Lymphedema Risk (%)'}
)

plt.title("Clinical Risk Matrix: BMI vs. Age", fontsize=15, pad=20)
plt.ylabel("BMI Category", fontsize=12)
plt.xlabel("Age Category", fontsize=12)
plt.tight_layout()

# Save the generated heatmap locally
plt.savefig("risk_matrix_bmi_age.png", dpi=300)
plt.close() 

# ---------------------------------------------------------
# ANALYSIS 2: TERMINAL SUMMARY REPORT
# ---------------------------------------------------------
print("="*50)
print(" CLINICAL RISK MATRIX RESULTS (%)")
print("="*50)
print(risk_matrix_pct.to_string(float_format="%.1f%%"))
print("\nDetailed Summary:")
print("-" * 50)

# Iterate through the matrix to print a highly readable text report
for index, row in risk_matrix_pct.iterrows():
    print(f"For patients in the '{index}' BMI category:")
    for age_group in age_labels:
        risk_val = row[age_group]
        print(f"  - Risk of lymphedema for age group {age_group}: {risk_val:.1f}%")
    print()