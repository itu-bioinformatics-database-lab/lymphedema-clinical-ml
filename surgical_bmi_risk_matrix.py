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

# Clean data: Convert to numeric formats and fill missing values with 0
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------------------------------------------------
# VARIABLES CONFIGURATION
# ---------------------------------------------------------
bmi_col = 'bmi' 
surgery_col = 'axillary_intervention' 
target_col = 'target'  # Target outcome (e.g., Lymphedema status: 0 or 1)

# 2. Data Preprocessing for Risk Matrix
# Define BMI bins and standardized clinical labels
bins = [0, 25, 30, 60]
labels = ['Normal (<25)', 'Overweight (25-30)', 'Obese (>30)']
df['BMI_Group'] = pd.cut(df[bmi_col], bins=bins, labels=labels)

# Calculate the mean of the target for each group intersection to get risk probability
# 'observed=False' prevents categorical grouping warnings in newer pandas versions
risk_matrix = df.groupby(['BMI_Group', surgery_col], observed=False)[target_col].mean().unstack()

# Map the surgery variables to descriptive columns (assuming 0 = SLNB, 1 = ALND)
risk_matrix.columns = ['SLNB (Low Intervention)', 'ALND (High Intervention)']

# Convert probabilities to percentages
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

plt.title("Clinical Risk Matrix: BMI vs. Surgical Aggressiveness", fontsize=15, pad=20)
plt.ylabel("BMI Category", fontsize=12)
plt.xlabel("Axillary Intervention Type", fontsize=12)
plt.tight_layout()

# Save the heatmap graph with high resolution
plt.savefig("risk_matrix.png", dpi=300)
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

# Iterate through the risk matrix to display a clean, automated textual report
for index, row in risk_matrix_pct.iterrows():
    print(f"For patients in the '{index}' category:")
    print(f"  - Risk of lymphedema with SLNB (Low Intervention): {row['SLNB (Low Intervention)']:.1f}%")
    print(f"  - Risk of lymphedema with ALND (High Intervention): {row['ALND (High Intervention)']:.1f}%")
    print()