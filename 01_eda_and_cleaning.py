# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:48:01 2026

@author: Hunter Denison
"""


# %% Cell 1 — Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

print("Libraries loaded successfully")

# Display settings
# Show all columns instead of truncating
pd.set_option('display.max_columns', None)

# Show full width in console
pd.set_option('display.width', None)

# Fix scientific notation — show 2 decimal places instead
pd.set_option('display.float_format', '{:,.2f}'.format)

# %% Cell 2 — Load the data
train = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train.csv', low_memory=False)
store = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/store.csv')

print("Train shape:", train.shape)
print("Store shape:", store.shape)

# %% Cell 3 — Examine Data
print(train.head(10))
print(store.head(10))

# %% Cell 4 — Column info and data types
print("\n--- TRAIN INFO ---")
train.info()

print("\n--- STORE INFO ---")
store.info()

# %% Cell 5 — Missing values
print("\n--- MISSING VALUES IN TRAIN ---")
print(train.isnull().sum())

print("\n--- MISSING VALUES IN STORE ---")
print(store.isnull().sum())

# %% Cell 6 — Summary statistics
print("\n--- TRAIN SUMMARY STATS ---")
print(train.describe())

# %% Cell 7 — Key counts
print("\nUnique stores in train:", train['Store'].nunique())
print("Unique stores in store file:", store['Store'].nunique())
print("Date range:", train['Date'].min(), "to", train['Date'].max())
print("Total rows:", len(train))

# %% Cell 8 — Sales distribution plot
plt.figure()
train[train['Sales'] > 0]['Sales'].hist(bins=50)
plt.title('Distribution of Daily Sales (excluding zeros)')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %% Cell 9 — How many days had zero sales?
zero_sales = train[train['Sales'] == 0]
print("\nRows with zero sales:", len(zero_sales))
print("Percentage of total:", round(len(zero_sales) / len(train) * 100, 2), "%")
print("\nZero sales only for closed stores?")
print(zero_sales['Open'].value_counts())

# %% Cell 10 — Investigating open stores with zero sales
open_zero = train[(train['Open'] == 1) & (train['Sales'] == 0)]

print("Open stores with zero sales:", len(open_zero))
print("\nOpen Stores with zeros")
print(open_zero['Store'].value_counts().head(20))

print("\nDay of Week investigation")
print(open_zero['DayOfWeek'].value_counts())

print("\nDate investigation")
print(open_zero['Date'].value_counts().head(10))

# %% Cell 11 — Store missing values with context
missing = store.isnull().sum()
missing_pct = (store.isnull().sum() / len(store) * 100).round(2)

missing_summary = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct,
    'Likely Meaning': [
        'N/A' if missing[col] == 0 
        else 'Possibly no competition or promo' 
        for col in store.columns
    ]
})

print(missing_summary[missing_summary['Missing Count'] > 0])

# %% Cell 12 — Examining open zero sales stores
open_zero = train[(train['Open'] == 1) & (train['Sales'] == 0)]

# Check if customers were also zero on these days
print("Customer counts on open/zero-sales days:")
print(open_zero['Customers'].describe())

print("\nRows where open, zero sales but customers > 0:")
print(len(open_zero[open_zero['Customers'] > 0]))

print("\nRows where open, zero sales AND zero customers:")
print(len(open_zero[open_zero['Customers'] == 0]))

# Check if these stores have normal sales on other days
problem_stores = open_zero['Store'].unique()
print("\nSample store with this issue - its full sales history:")
# Checked one through five
sample_store = problem_stores[0]
print(train[train['Store'] == sample_store].sort_values('Date').head(20))
print(train[train['Store'] == sample_store].sort_values(by = ['Sales','Open'], ascending = [True, False]).head(20))

#54 records show stores marked as open with zero sales and zero customers. 
#No consistent pattern was found across day of week, promotional status, school holidays, or store type. 
#These records are treated as data entry errors and will be excluded from modeling.
