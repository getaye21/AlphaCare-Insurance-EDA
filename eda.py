# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read with tab delimiter
df = pd.read_csv('ethiopia_insurance_data.csv', sep='\t', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print("Columns found:", df.columns.tolist())
print("Shape:", df.shape)

# Convert columns
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
df['DriverAge'] = pd.to_numeric(df['DriverAge'], errors='coerce')

# Add time features
df['Year'] = df['TransactionMonth'].dt.year
df['Month'] = df['TransactionMonth'].dt.month

# Create output folder
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

# 4. Low‑risk segments – improved for small dataset
# Group by PostalCode, LegalType, Gender
low_risk = df.groupby(['PostalCode', 'LegalType', 'Gender']).agg(
    AvgLossRatio=('LossRatio', 'mean'),
    PolicyCount=('PolicyID', 'count'),
    AvgPremium=('TotalPremium', 'mean')
).reset_index()

# Show all groups before filtering
print("\nAll groups (sorted by AvgLossRatio):")
print(low_risk.sort_values('AvgLossRatio'))

# Use a threshold: bottom 30% by loss ratio (or any group with loss ratio <= 0.1)
threshold = low_risk['AvgLossRatio'].quantile(0.3)  # 30th percentile
targets = low_risk[(low_risk['AvgLossRatio'] <= threshold) & (low_risk['PolicyCount'] >= 1)]

# Also include any group with zero loss ratio (safe)
zero_loss = low_risk[low_risk['AvgLossRatio'] == 0]
targets = pd.concat([targets, zero_loss]).drop_duplicates()

targets = targets.sort_values('AvgLossRatio')
print("\n🔍 Low‑risk target segments (bottom 30% by loss ratio or zero loss):")
print(targets)

# Save to CSV
targets.to_csv('outputs/low_risk_segments.csv', index=False)

print("\n✅ Results saved in 'outputs/' folder")
