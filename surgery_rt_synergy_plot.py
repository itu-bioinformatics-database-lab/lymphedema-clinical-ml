import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('final_dataset.csv')

# Group the combination of Surgery (ALND) and Radiotherapy (RT)
def combo_risk(row):
    if row['axillary_intervention'] == 1 and row['axillary_radiotherapy'] == 1:
        return "ALND + RT\n(Highest Burden)"
    elif row['axillary_intervention'] == 1:
        return "ALND Only"
    elif row['axillary_radiotherapy'] == 1:
        return "RT without ALND"
    else:
        return "Low Intervention\n(SLNB)"

df['Intervention_Group'] = df.apply(combo_risk, axis=1)

# Calculate the mean and sort values to display ascending bars
combo_stats = df.groupby('Intervention_Group')['target'].mean().sort_values() * 100

# Terminal Output of the Analysis Results
print("\n--- Lymphedema Rates (By Intervention Group) ---")
for group, rate in combo_stats.items():
    # Replace newline characters with spaces for a cleaner console format
    group_name_clean = group.replace('\n', ' ')
    print(f"{group_name_clean}: {rate:.1f}%")
print("--------------------------------------------------\n")

plt.figure(figsize=(10, 6))

# Professional gradient blue color palette indicating progressive risk
professional_colors = ['#A6C4E1', '#749CBE', '#42749B', '#114C78']

ax = combo_stats.plot(kind='bar', color=professional_colors)

# Axis Configuration
plt.ylabel("Lymphedema Rate (%)")
plt.xlabel("")  # Kept blank for a cleaner look since bar labels are descriptive
plt.xticks(rotation=0)

# Add percentages on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontweight='bold', color='#333333')

# Save locally with high resolution (300 DPI)
plt.savefig("surgery_rt_synergy.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plot successfully saved as 'surgery_rt_synergy.png'.")