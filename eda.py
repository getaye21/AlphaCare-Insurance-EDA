# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('ethiopia_insurance_data.csv', parse_dates=['TransactionMonth'])
print("Shape:", df.shape)

# Basic cleaning
df['DriverAge'] = pd.to_numeric(df['DriverAge'], errors='coerce')

# Add time features
df['Year'] = df['TransactionMonth'].dt.year
df['Month'] = df['TransactionMonth'].dt.month

# Create output folder if needed (optional)
import os
os.makedirs('outputs', exist_ok=True)

# 1. Univariate plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['TotalPremium'], kde=True, ax=axes[0,0])
axes[0,0].set_title('Premium Distribution')
sns.histplot(df['TotalClaims'], discrete=True, ax=axes[0,1])
axes[0,1].set_title('Claim Count')
sns.histplot(df['DriverAge'].dropna(), kde=True, ax=axes[1,0])
axes[1,0].set_title('Driver Age')
sns.histplot(df['LossRatio'], kde=True, ax=axes[1,1])
axes[1,1].set_title('Loss Ratio')
plt.tight_layout()
plt.savefig('outputs/univariate.png')
plt.close()

# 2. Premium vs Claim Amount
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='TotalPremium', y='ClaimAmount', alpha=0.6)
plt.title('Premium vs Claim Amount')
plt.savefig('outputs/premium_vs_claim.png')
plt.close()

# 3. Average Loss Ratio by PostalCode
postal_agg = df.groupby('PostalCode')['LossRatio'].mean().sort_values()
plt.figure(figsize=(10,4))
postal_agg.plot(kind='bar')
plt.title('Average Loss Ratio by Postal Code')
plt.xticks(rotation=45)
plt.savefig('outputs/loss_by_postal.png')
plt.close()

# 4. Low‑risk segments (LossRatio < median and at least 2 policies)
low_risk = df.groupby(['PostalCode', 'LegalType', 'Gender']).agg(
    AvgLossRatio=('LossRatio', 'mean'),
    PolicyCount=('PolicyID', 'count')
).reset_index()
median_loss = low_risk['AvgLossRatio'].median()
targets = low_risk[(low_risk['AvgLossRatio'] < median_loss) & (low_risk['PolicyCount'] >= 2)]
targets = targets.sort_values('AvgLossRatio')
print("\n🔍 Low‑risk target segments:\n", targets)

# Save to CSV
targets.to_csv('outputs/low_risk_segments.csv', index=False)
print("\n✅ Results saved in 'outputs/' folder")
