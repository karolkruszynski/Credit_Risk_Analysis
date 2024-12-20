# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_csv('credit_risk_data.csv')
print(df.head(5))
print(df.columns)

# Dropping Unnamed: 0 column
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.columns)

### Analysis of the distribution of loan amounts by purpose ###

# Data Preparation
print(df.isna().sum())

# Let's drop rows /w NA values in credit_history and employment columns
df.dropna(axis=0,inplace=True)
print(df.isna().sum())

# Now we can start our analysis
# Grouped data by purpose
purpose_data = df.groupby('purpose')
print(purpose_data.describe())

# Count credit by purpose
print(purpose_data['purpose'].value_counts())
# Most credit was taken for business purpose, fewest for retraining and other purpose.

# Mean, min, max credit amount by purpose distribution
print(purpose_data['credit_amount'].mean().sort_values(ascending=False))
print(purpose_data['credit_amount'].max().sort_values(ascending=False))
print(purpose_data['credit_amount'].min().sort_values(ascending=False))
# We can see some issue here because purpose - other have same amount in every describe type mean, max, min
# Iterate through groups and select groups with 'other'
for name, group in df.groupby('purpose'):
    if 'other' in name:
        print(group['credit_amount'])
# As we can see there is only 1 row of data with purpose = other with credit_amount = 18242
# This data misleads us soo we delete this row
df = df[~df['purpose'].str.contains('other',na=False)]
purpose_data = df.groupby('purpose')
print(purpose_data['purpose'].unique())

# Visualize data of mean, max, min amount of credit by purpose

# MEAN AMOUNT OF CREDIT BY PURPOSE
# Create a data with mean values by purpose
mean_credit_amount = df.groupby('purpose')['credit_amount'].mean().sort_values(ascending=False)
# Sorted data with index purpose searched by previous data .loc[]
sorted_data = df.set_index('purpose').loc[mean_credit_amount.index]

# Creating plot
plt.figure(figsize=(12, 6))
sorted_data.boxplot(column='credit_amount', by='purpose', vert=False, grid=True, patch_artist=True)

# Plot Adjustments
plt.title('Distribution of loan amounts by purpose')
plt.suptitle('')
plt.xlabel('Credit amount')
plt.ylabel('Purpose of the credit')
plt.tight_layout()

# Show plot
plt.show()

### Effect of age and length of credit on credit quality ###
# age, duration (in years),class - credit quality

credit_quality = df.groupby('class')
print(credit_quality.describe())
print(credit_quality[['age','duration']].value_counts().loc['good'])