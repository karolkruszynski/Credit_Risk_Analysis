# IMPORTS
import pandas as pd
import numpy as np
import matplotlib as plot

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