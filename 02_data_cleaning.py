# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:51:16 2026

@author: Hunter Denison
"""

# %% Cell 1 — Imports and load data
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

train = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train.csv', low_memory=False)
store = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/store.csv')

print("Raw train shape:", train.shape)
print("Raw store shape:", store.shape)

# %% Cell 2 — Remove closed store days
# Stores that are closed have no predictive value for sales forecasting
# We only want to model days when stores are actually open
train_open = train[train['Open'] == 1].copy()
print("After removing closed days:", train_open.shape)
print("Rows removed:", len(train) - len(train_open))

# %% Cell 3 — Remove open stores with zero sales
# As established in EDA: 54 records show open stores with zero sales
# and zero customers. No consistent pattern found — treated as data errors.
train_clean = train_open[train_open['Sales'] > 0].copy()
print("After removing zero sales anomalies:", train_clean.shape)
print("Rows removed:", len(train_open) - len(train_clean))

# %% Cell 4 — Convert Date to datetime
train_clean['Date'] = pd.to_datetime(train_clean['Date'])
print("\nDate dtype after conversion:", train_clean['Date'].dtype)
print("Date range:", train_clean['Date'].min(), "to", train_clean['Date'].max())

# %% Cell 5 — Handle store file missing values
# Review what's missing
print("\nStore missing values before cleaning:")
print(store.isnull().sum())

# CompetitionDistance — missing likely means no nearby competitor
# Filling with a large number to represent very distant/no competition
# Can't fill with 0 because it communicates opposite intended effect
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(
    store['CompetitionDistance'].max() * 2
)

# Competition open since; Missing means no competition
# Filling month and year with 0 to indicate no competitor
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0)

# Promo2 since; missing means not participating in Promo2
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0)

# PromoInterval; missing means no promo interval
store['PromoInterval'] = store['PromoInterval'].fillna('None')

print("\nStore missing values after cleaning:")
print(store.isnull().sum())

# %% Cell 6 — Merge train and store data
df = pd.merge(train_clean, store, on='Store', how='left')
print("\nMerged dataframe shape:", df.shape)
print("Any nulls after merge?")
print(df.isnull().sum()[df.isnull().sum() > 0])

# %% Cell 7 — Verify merge
print("\nSample of merged data:")
print(df.head())

# %% Cell 8 — Save cleaned data
df.to_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train_cleaned.csv', index=False)
print("\nCleaned data saved to ../data/train_cleaned.csv")
print("Final clean dataset shape:", df.shape)