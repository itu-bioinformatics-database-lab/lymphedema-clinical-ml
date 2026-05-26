import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Set publication-quality plot parameters via rcParams
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def run_lymphedema_analysis(file_path='final_dataset.csv'):
    # 1. Load Dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Standardize column names
    df.columns = df.columns.str.strip()
    
    if 'removed_lymph_nodes' not in df.columns or 'target' not in df.columns:
        print("Error: Missing required columns 'removed_lymph_nodes' or 'target'.")
        return

    # Data Cleaning
    df = df.dropna(subset=['removed_lymph_nodes', 'target'])
    df['removed_lymph_nodes'] = df['removed_lymph_nodes'].astype(float)
    df['target'] = df['target'].astype(int)

    # 2. Model Training & Exact AUC Calculation
    threshold = 5
    X = df[['removed_lymph_nodes']]
    y = df['target']
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)
    y_probs = clf.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_probs)

    # 3. Academic Visualization (Publication Ready)
    sns.set_theme(style="white")
    
    # Calculate empirical risk per unique node count
    empirical_data = df.groupby('removed_lymph_nodes')['target'].agg(['mean', 'count']).reset_index()
    
    # Scatter plot for empirical risk distribution
    plt.scatter(
        empirical_data['removed_lymph_nodes'], 
        empirical_data['mean'], 
        s=empirical_data['count'] * 15, 
        color='#2c3e50', 
        alpha=0.5, 
        edgecolors='black',
        label='Empirical Risk (Size $\propto$ Sample Size)'
    )
    
    # Fit and overlay Logistic Regression Trend Line
    try:
        import statsmodels
        sns.regplot(
            x='removed_lymph_nodes', 
            y='target', 
            data=df, 
            logistic=True, 
            scatter=False, 
            ci=95, 
            color='#e74c3c', 
            line_kws={'linewidth': 2.5},
            label=f'Logistic Regression Fit\n(ROC AUC: {auc_score:.4f})'
        )
    except ImportError:
        # Fallback to second-order polynomial smooth trend if statsmodels is not installed
        sns.regplot(
            x='removed_lymph_nodes', 
            y='target', 
            data=df, 
            scatter=False, 
            order=2,
            color='#e74c3c', 
            line_kws={'linewidth': 2.5},
            label='Trend Fit (Polynomial Order 2)'
        )

    # Vertical line indicating the clinical threshold
    plt.axvline(
        x=threshold, 
        color='#7f8c8d', 
        linestyle='--', 
        linewidth=2, 
        label=f'Critical Threshold ({threshold} Nodes)'
    )
    
    # Axes Configurations
    plt.xlabel('Number of Removed Lymph Nodes', labelpad=10)
    plt.ylabel('Probability of Lymphedema', labelpad=10)
    plt.ylim(-0.05, 1.05)
    
    # Academic Despining and Grids
    sns.despine(top=True, right=True)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Legend setup positioned outside on the upper right
    plt.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        frameon=True, 
        facecolor='white', 
        edgecolor='none',
        labelspacing=1.2
    )
    
    # Save the figure as 300 DPI directly to the current directory
    output_filename = 'lymphedema_risk_analysis_plot.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success: Plot successfully saved as '{output_filename}' at 300 DPI.")

if __name__ == "__main__":
    run_lymphedema_analysis()