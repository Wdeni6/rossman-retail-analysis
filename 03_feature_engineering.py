# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:01:08 2026

@author: Hunter Denison
"""

# %% Cell 1 — Imports and load cleaned data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

df = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train_cleaned.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

print("Loaded shape:", df.shape)
print(df.head())

# %% Cell 2 — Basic date features
# Breaking date into components gives the model calendar awareness
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Date'].dt.quarter

print("Date features added")
print(df[['Date', 'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter']].head(10))

# %% Cell 3 — Beginning, middle, end of month
df['IsMonthStart'] = (df['Day'] <= 5).astype(int)
df['IsMonthEnd'] = (df['Day'] >= 25).astype(int)

print("\nMonth position features:")
print(df[['Day', 'IsMonthStart', 'IsMonthEnd']].head(10))

# %% Cell 4 — Competition tenure
# Longer tenure = more established competition pressure
df['CompetitionTenureMonths'] = (
    (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
    (df['Month'] - df['CompetitionOpenSinceMonth'])
)

# If competition year is 0 it means no competitor; set tenure to 0
df['CompetitionTenureMonths'] = df['CompetitionTenureMonths'].clip(lower=0)
df.loc[df['CompetitionOpenSinceYear'] == 0, 'CompetitionTenureMonths'] = 0

print("\nCompetition tenure sample:")
print(df[['Store', 'Date', 'CompetitionOpenSinceYear', 
          'CompetitionOpenSinceMonth', 'CompetitionTenureMonths']].head(10))

# %% Cell 5 — Is competition new?
# New competitors (opened in last 6 months) may have stronger impact
df['IsNewCompetitor'] = (
    (df['CompetitionTenureMonths'] > 0) & 
    (df['CompetitionTenureMonths'] <= 6)
).astype(int)

print("\nNew competitor flag counts:")
print(df['IsNewCompetitor'].value_counts())

# %% Cell 6 — Promo2 participation duration
# How long has the store been running the continuous promo?
df['Promo2TenureMonths'] = (
    (df['Year'] - df['Promo2SinceYear']) * 12 +
    (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.33
)

df['Promo2TenureMonths'] = df['Promo2TenureMonths'].clip(lower=0)
df.loc[df['Promo2SinceYear'] == 0, 'Promo2TenureMonths'] = 0

print("\nPromo2 tenure sample:")
print(df[['Store', 'Promo2', 'Promo2SinceYear', 
          'Promo2TenureMonths']].head(10))

# %% Cell 7 — Store type and assortment encoding
store_type_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
assortment_map = {'a': 1, 'b': 2, 'c': 3}

df['StoreTypeEncoded'] = df['StoreType'].map(store_type_map)
df['AssortmentEncoded'] = df['Assortment'].map(assortment_map)

print("\nStore type distribution:")
print(df['StoreType'].value_counts())
print("\nAssortment distribution:")
print(df['Assortment'].value_counts())

# %% Cell 8 — Sales per customer
# Avg customer spend per visit
# This is a revenue efficiency metric
df['SalesPerCustomer'] = df['Sales'] / df['Customers']

print("\nSales per customer stats:")
print(df['SalesPerCustomer'].describe())

# %% Cell 9 — Visualize key engineered features

# Sales by month
plt.figure()
df.groupby('Month')['Sales'].mean().plot(kind='bar', color='steelblue')
plt.title('Average Daily Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Sales by day of week
plt.figure()
df.groupby('DayOfWeek')['Sales'].mean().plot(kind='bar', color='coral')
plt.title('Average Daily Sales by Day of Week (1=Monday)')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Impact of promotions
plt.figure()
df.groupby('Promo')['Sales'].mean().plot(kind='bar', color='green')
plt.title('Average Sales: No Promo vs Promo')
plt.xlabel('Promo (0=No, 1=Yes)')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Competition distance vs sales
plt.figure()
plt.scatter(df['CompetitionDistance'], df['Sales'], alpha=0.1, s=1)
plt.title('Competition Distance vs Sales')
plt.xlabel('Competition Distance')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# Sales per customer distribution
plt.figure()
df['SalesPerCustomer'].hist(bins=50, color='purple')
plt.title('Distribution of Sales Per Customer')
plt.xlabel('Sales Per Customer')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %% Cell 10 — Correlation of new features with sales
feature_cols = [
    'Promo', 'DayOfWeek', 'Month', 'Quarter',
    'IsMonthStart', 'IsMonthEnd', 'CompetitionDistance',
    'CompetitionTenureMonths', 'IsNewCompetitor',
    'Promo2', 'Promo2TenureMonths', 'StoreTypeEncoded',
    'AssortmentEncoded', 'SchoolHoliday', 'StateHoliday'
]

# StateHoliday needs to be numeric first
df['StateHolidayEncoded'] = (df['StateHoliday'] != '0').astype(int)
feature_cols[-1] = 'StateHolidayEncoded'

correlations = df[feature_cols + ['Sales']].corr()['Sales'].drop('Sales')
correlations_sorted = correlations.abs().sort_values(ascending=False)

print("\nFeature correlations with Sales (absolute value, sorted):")
print(correlations_sorted)

plt.figure(figsize=(10, 6))
correlations_sorted.plot(kind='bar', color='steelblue')
plt.title('Feature Correlation with Sales')
plt.ylabel('Absolute Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% Cell 11 — Lag features and rolling averages
# Sort by store and date first for time series features
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

# Lag features
df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
df['Sales_Lag14'] = df.groupby('Store')['Sales'].shift(14)
df['Sales_Lag30'] = df.groupby('Store')['Sales'].shift(30)

# Rolling averages to smooth out day-to-day noise
# 7 day rolling mean = recent weekly trend
df['Sales_Rolling7Mean'] = (
    df.groupby('Store')['Sales']
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

# 30 day rolling mean = longer term trend
df['Sales_Rolling30Mean'] = (
    df.groupby('Store')['Sales']
    .transform(lambda x: x.shift(1).rolling(30).mean())
)

# Rolling std captures volatility in a store's sales
df['Sales_Rolling7Std'] = (
    df.groupby('Store')['Sales']
    .transform(lambda x: x.shift(1).rolling(7).std())
)

print("Lag and rolling features added")
print("\nMissing values from lag features:")
print(df[['Sales_Lag7', 'Sales_Lag14', 'Sales_Lag30',
          'Sales_Rolling7Mean', 'Sales_Rolling30Mean',
          'Sales_Rolling7Std']].isnull().sum())

# %% Cell 12 — Drop rows with nulls from lag features
# Early rows for each store won't have enough history
# for lag/rolling features we will remove them
df_model = df.dropna(subset=[
    'Sales_Lag7', 'Sales_Lag14', 'Sales_Lag30',
    'Sales_Rolling7Mean', 'Sales_Rolling30Mean',
    'Sales_Rolling7Std'
]).copy()

print("\nShape after dropping lag nulls:", df_model.shape)
print("Rows removed:", len(df) - len(df_model))

# %% Cell 13 — Save model-ready dataset
df_model.to_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train_model_ready.csv', index=False)
print("\nModel-ready dataset saved")
print("Final shape:", df_model.shape)
print("Total features:", len(df_model.columns))
print("\nAll columns:")
for col in df_model.columns:
    print(" -", col)