import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Preparation
try:
    df = pd.read_csv('final_dataset.csv')
    cols = ['axillary_intervention', 'axillary_radiotherapy', 'chemotherapy', 'target']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- INDIVIDUAL RISK CALCULATION ---
# Calculate lymphedema rates for each distinct intervention
risk_alnd = df[df['axillary_intervention'] == 1]['target'].mean() * 100
risk_rt = df[df['axillary_radiotherapy'] == 1]['target'].mean() * 100
risk_chemo = df[df['chemotherapy'] == 1]['target'].mean() * 100

# 2. Create Clinical Treatment Burden Score
df['Treatment_Burden'] = (
    df['axillary_intervention'] + 
    df['axillary_radiotherapy'] + 
    df['chemotherapy']
)

# 3. Visualization
plt.figure(figsize=(12, 8), dpi=300)

# Apply a clean, professional theme
sns.set_theme(style="whitegrid")

burden_stats = df.groupby('Treatment_Burden')['target'].mean() * 100
colors = sns.color_palette("Reds", n_colors=len(burden_stats))

ax = sns.barplot(
    x=burden_stats.index, 
    y=burden_stats.values, 
    palette=colors, 
    edgecolor="#2C3E50", 
    linewidth=1.2
)

# Add percentage values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                color='#2C3E50', xytext=(0, 8),
                textcoords='offset points')

# --- INFO BOX (Placed Top-Right) ---
burden_definition = (
    "INTERVENTION-BASED INDIVIDUAL RISKS:\n\n"
    f"• Axillary Lymph Node Dissection (ALND): {risk_alnd:.1f}%\n"
    f"• Axillary Radiotherapy: {risk_rt:.1f}%\n"
    f"• Chemotherapy History: {risk_chemo:.1f}%\n"
    "----------------------------------------------------\n"
    "SCORE CALCULATION:\n"
    "Each intervention adds +1 point to the score.\n"
    "The chart displays the cumulative effect of these risks."
)

# Place text box in the top-right corner of the plot
plt.text(0.96, 0.96, burden_definition, transform=ax.transAxes, 
         fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.95, edgecolor='#B0BEC5', boxstyle='round,pad=0.8'))

# Axis Adjustments
plt.xlabel("Total Treatment Burden Score (0 - 3)", labelpad=15, color='#2C3E50')
plt.ylabel("Lymphedema Rate in Group (%)", labelpad=15, color='#2C3E50')

plt.yticks(np.arange(0, 101, 10), fontsize=11) 
plt.xticks(fontsize=12)
plt.ylim(0, 110) # Provides extra vertical space at the top for the info box

# Clean up chart borders
sns.despine(left=True)
plt.tight_layout()

# Save the plot with high resolution
plt.savefig("treatment_burden_detailed_risks.png", bbox_inches='tight', dpi=300)
plt.show()

# Terminal Outputs
print("Analysis completed successfully. Individual risks calculated:")
print(f"ALND: {risk_alnd:.1f}% | RT: {risk_rt:.1f}% | Chemo: {risk_chemo:.1f}%")