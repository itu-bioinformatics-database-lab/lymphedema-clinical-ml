import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
df = pd.read_csv('final_dataset.csv')
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. Analysis: Molecular Subtypes and Risk (St. Gallen Criteria)
def categorize_subtype(row):
    er_pr_pos = (row['er_percent'] > 0) or (row['pr_percent'] > 0)
    her2_pos = row['biomarker'] == 1
    ki67_high = row['ki67_percent'] >= 20
    
    if er_pr_pos:
        if not her2_pos and not ki67_high: 
            return "Luminal A\n(ER/PR+, HER2-, Ki67<20%)"
        else: 
            return "Luminal B\n(ER/PR+, HER2+ or Ki67>=20%)"
    else:
        if her2_pos: 
            return "HER2 Positive\n(ER-, PR-, HER2+)"
        else: 
            return "Triple Negative\n(ER-, PR-, HER2-)"

# Apply categorization
df['molecular_subtype'] = df.apply(categorize_subtype, axis=1)

# Calculate prevalence and counts
stats_subtype = df.groupby('molecular_subtype')['target'].agg(['mean', 'count']).reset_index()
stats_subtype['mean'] *= 100

# --- TERMINAL REPORT GENERATION ---
print("="*60)
print("MOLECULAR SUBTYPE CATEGORIZATION AND RISK ANALYSIS REPORT")
print("="*60)
print(f"Total Evaluated Records: {len(df)}")
print("-" * 60)

# Sort results by risk rate in descending order for reporting and plotting
stats_sorted = stats_subtype.sort_values('mean', ascending=False)

for index, row in stats_sorted.iterrows():
    # Replace newline characters with spaces for a cleaner console layout
    subtype_name = row['molecular_subtype'].replace('\n', ' ')
    count = int(row['count'])
    risk = row['mean']
    
    print(f"Molecular Subtype: {subtype_name}")
    print(f"  -> Sample Count : {count}")
    print(f"  -> Risk Rate    : {risk:.1f}%")
    print("-" * 60)
# ---------------------------------------------------------

# 3. Visualization
plt.figure(figsize=(12, 7), dpi=150)
sns.set_style("whitegrid")

# Create bar plot using the sorted data
ax = sns.barplot(
    x='molecular_subtype', 
    y='mean', 
    data=stats_sorted, 
    palette='YlOrRd_r'
)

# Add percentage annotations on top of the bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}%", 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', 
        va='bottom', 
        fontweight='bold'
    )

# Formatting labels and limits
plt.ylim(0, 100)  # Sets y-axis range from 0 to 100
plt.xlabel("Molecular Subtype", fontsize=12)
plt.ylabel("Risk Rate (%)", fontsize=12)

# Save directly to the current directory and close the plot
plt.savefig("molecular_subtype_risk.png", bbox_inches='tight') 
plt.close()

print("Plot successfully saved as 'molecular_subtype_risk.png'.")