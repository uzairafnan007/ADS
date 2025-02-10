import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load dataset (replace with your actual data source)
df = pd.read_csv("clv_data.csv")

# --- Data Preparation ---
# Introduce 20% missing values in numeric columns
np.random.seed(42)
numeric_cols = df.select_dtypes(include=np.number).columns
missing_mask = np.random.rand(*df[numeric_cols].shape) < 0.2
df_missing = df.copy()
df_missing[numeric_cols] = df_missing[numeric_cols].mask(missing_mask)

# --- Imputation ---
# Create imputation strategies
strategies = {
    'Mean': 'mean',
    'Median': 'median', 
    'Mode': 'most_frequent'
}

imputed_dfs = {}
for name, strategy in strategies.items():
    imputer = SimpleImputer(strategy=strategy)
    imputed = imputer.fit_transform(df_missing[numeric_cols])
    imputed_dfs[name] = pd.DataFrame(imputed, columns=numeric_cols)

# --- Enhanced Visualization ---
plt.figure(figsize=(15, 10))

# 1. Missing Value Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(df_missing[numeric_cols].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Pattern")

# 2. Distribution Comparison
plt.subplot(2, 2, 2)
sns.kdeplot(df[numeric_cols[0]], label='Original', linewidth=3)
for color, (name, df_imputed) in zip(['red', 'green', 'blue'], imputed_dfs.items()):
    sns.kdeplot(df_imputed[numeric_cols[0]], label=f'{name} Imputed', linestyle='--', color=color)
plt.title("Distribution Comparison")
plt.legend()

# 3. Boxplot Comparison
plt.subplot(2, 2, 3)
plot_data = pd.concat([
    df[numeric_cols[0]].rename('Original'),
    imputed_dfs['Mean'][numeric_cols[0]].rename('Mean'),
    imputed_dfs['Median'][numeric_cols[0]].rename('Median'),
    imputed_dfs['Mode'][numeric_cols[0]].rename('Mode')
], axis=1)
sns.boxplot(data=plot_data.melt(), x='variable', y='value')
plt.title("Value Distribution Comparison")
plt.xticks(rotation=45)

# 4. Missing Value Counts
plt.subplot(2, 2, 4)
missing_counts = df_missing[numeric_cols].isnull().sum()
sns.barplot(x=missing_counts.index, y=missing_counts.values)
plt.title("Missing Values per Column")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Add this after creating df_mode_imputed but before visualization

# Create combined dataframe with original and imputed values
df_combined = df_missing.copy()
for col in numeric_cols:
    df_combined[f'{col}_mean'] = imputed_dfs['Mean'][col]
    df_combined[f'{col}_median'] = imputed_dfs['Median'][col]
    df_combined[f'{col}_mode'] = imputed_dfs['Mode'][col]

# Print the combined dataset
print("\nCombined Dataset with Original and Imputed Values:")
pd.set_option('display.max_columns', None)  # Show all columns
print(df_combined.head())
pd.reset_option('display.max_columns')  # Reset display options