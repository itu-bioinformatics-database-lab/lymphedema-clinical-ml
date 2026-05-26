import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preparation
df = pd.read_csv('final_dataset.csv')

# 2. Calculate Chemotherapy Groups and Risks
# Mapping values to readable group names for clean plot labels and legends
df['Chemotherapy Status'] = df['chemotherapy'].map({1.0: 'Chemo Received (+)', 0.0: 'No Chemo (-)'})
df['Surgery Type'] = df['axillary_intervention'].map({1.0: 'ALND (Dissection)', 0.0: 'SLNB (Biopsy)'})

# Statistical Summary Calculation
chemo_risk = df.groupby('Chemotherapy Status')['target'].mean() * 100
diff = chemo_risk['Chemo Received (+)'] - chemo_risk['No Chemo (-)']

# 3. Visualization: Chemotherapy and Surgery Interaction Plot
plt.figure(figsize=(12, 8), dpi=300)
sns.set_style("whitegrid")

# Visualizing how chemotherapy interacts with surgery type to affect risk
# Added dodge=0.1 to prevent overlapping lines and error bars
ax = sns.pointplot(x='Surgery Type', y='target', hue='Chemotherapy Status', data=df,
                   palette={'Chemo Received (+)': '#C0392B', 'No Chemo (-)': '#2980B9'},
                   markers=["D", "o"], linestyles=["-", "--"], capsize=.1, dodge=0.1)

# Plot Labeling & Styling
plt.ylabel("Lymphedema Probability (%)", fontsize=14)
plt.xlabel("Axillary Surgery Approach", fontsize=14)

# Format y-axis ticks as percentages safely
ticks = plt.gca().get_yticks()
plt.gca().set_yticks(ticks)
plt.gca().set_yticklabels(['{:,.0f}%'.format(x*100) for x in ticks])

# Ensure the legend title is clean
plt.legend(title='Chemotherapy Status')

sns.despine()

# Save the plot with high resolution (300 DPI)
plt.savefig("chemo_interaction_analysis.png", bbox_inches='tight', dpi=300)
plt.show()

# Terminal Outputs
print("--- ANALYSIS RESULTS ---")
print(f"Risk for Chemo Received (+): {chemo_risk['Chemo Received (+)']:.1f}%")
print(f"Risk for No Chemo (-): {chemo_risk['No Chemo (-)']:.1f}%")
print(f"Chemotherapy Standalone Risk Increase (Difference): {diff:.1f}%")