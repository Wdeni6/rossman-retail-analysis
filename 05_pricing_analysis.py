# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:55:58 2026

@author: Hunter Denison
"""

# %% Cell 1 — Imports and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

df = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train_model_ready.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

print("Loaded shape:", df.shape)

# %% Cell 2 — Promo lift analysis overall
# How much does a promotion increase sales on average?
promo_summary = df.groupby('Promo')['Sales'].agg(['mean', 'median', 'std', 'count'])
promo_summary.index = ['No Promo', 'Promo']
print("Sales summary by promo status:")
print(promo_summary)

promo_lift = (promo_summary.loc['Promo', 'mean'] - 
              promo_summary.loc['No Promo', 'mean']) / \
              promo_summary.loc['No Promo', 'mean'] * 100
print(f"\nOverall promo lift: {promo_lift:.1f}%")

plt.figure()
df.groupby('Promo')['Sales'].mean().plot(
    kind='bar', color=['steelblue', 'coral'], 
    width=0.5
)
plt.title('Average Sales: No Promo vs Promo')
plt.xticks([0, 1], ['No Promo', 'Promo'], rotation=0)
plt.ylabel('Average Sales')
plt.tight_layout()
plt.show()

# %% Cell 3 — Statistical significance of promo lift
# Does promo actually matter or could this be random chance?
promo_sales = df[df['Promo'] == 1]['Sales']
no_promo_sales = df[df['Promo'] == 0]['Sales']

t_stat, p_value = stats.ttest_ind(promo_sales, no_promo_sales)
print("\nStatistical test for promo effect:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
if p_value < 0.05:
    print("Result: Promo effect is statistically significant (p < 0.05)")
else:
    print("Result: Promo effect is NOT statistically significant")

# %% Cell 4 — Promo lift by store type
# Does promo work equally well for all store types?
promo_by_store = df.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
promo_by_store.columns = ['No Promo', 'Promo']
promo_by_store['Lift_Pct'] = (
    (promo_by_store['Promo'] - promo_by_store['No Promo']) / 
    promo_by_store['No Promo'] * 100
).round(1)

print("\nPromo lift by store type:")
print(promo_by_store)

plt.figure()
promo_by_store[['No Promo', 'Promo']].plot(
    kind='bar', color=['steelblue', 'coral']
)
plt.title('Promo vs No Promo Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.xticks(rotation=0)
plt.legend(['No Promo', 'Promo'])
plt.tight_layout()
plt.show()

# %% Cell 5 — Promo lift by day of week
# Are some days more responsive to promotions than others?
promo_by_day = df.groupby(['DayOfWeek', 'Promo'])['Sales'].mean().unstack()
promo_by_day.columns = ['No Promo', 'Promo']
promo_by_day['Lift_Pct'] = (
    (promo_by_day['Promo'] - promo_by_day['No Promo']) / 
    promo_by_day['No Promo'] * 100
).round(1)

print("\nPromo lift by day of week:")
print(promo_by_day)

plt.figure()
promo_by_day['Lift_Pct'].plot(kind='bar', color='green')
plt.title('Promo Lift % by Day of Week (1=Monday)')
plt.xlabel('Day of Week')
plt.ylabel('Lift %')
plt.xticks(rotation=0)
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# %% Cell 6 — Promo lift by month
# Are promotions more effective in certain months?
promo_by_month = df.groupby(['Month', 'Promo'])['Sales'].mean().unstack()
promo_by_month.columns = ['No Promo', 'Promo']
promo_by_month['Lift_Pct'] = (
    (promo_by_month['Promo'] - promo_by_month['No Promo']) / 
    promo_by_month['No Promo'] * 100
).round(1)

print("\nPromo lift by month:")
print(promo_by_month)

plt.figure()
promo_by_month['Lift_Pct'].plot(kind='bar', color='purple')
plt.title('Promo Lift % by Month')
plt.xlabel('Month')
plt.ylabel('Lift %')
plt.xticks(rotation=0)
plt.axhline(promo_lift, color='red', linestyle='--', 
            label=f'Overall average: {promo_lift:.1f}%')
plt.legend()
plt.tight_layout()
plt.show()

# %% Cell 7 — Competition distance impact on sales
# Do stores with closer competitors perform worse?
# Create distance buckets for cleaner analysis
df['DistanceBucket'] = pd.cut(
    df['CompetitionDistance'],
    bins=[0, 500, 1000, 2000, 5000, 10000, float('inf')],
    labels=['<500m', '500m-1km', '1-2km', '2-5km', '5-10km', '>10km']
)

distance_sales = df.groupby('DistanceBucket')['Sales'].mean()
print("\nAverage sales by competition distance:")
print(distance_sales)

plt.figure()
distance_sales.plot(kind='bar', color='steelblue')
plt.title('Average Sales by Competition Distance')
plt.xlabel('Distance to Nearest Competitor')
plt.ylabel('Average Sales')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# %% Cell 8 — New competitor impact
# What happens to sales when a new competitor opens nearby?
new_comp_summary = df.groupby('IsNewCompetitor')['Sales'].agg(
    ['mean', 'median', 'count']
)
new_comp_summary.index = ['No New Competitor', 'New Competitor Nearby']
print("\nSales impact of new competitor:")
print(new_comp_summary)

new_comp_impact = (
    new_comp_summary.loc['New Competitor Nearby', 'mean'] -
    new_comp_summary.loc['No New Competitor', 'mean']
) / new_comp_summary.loc['No New Competitor', 'mean'] * 100
print(f"\nNew competitor sales impact: {new_comp_impact:.1f}%")

# %% Cell 9 — Promo2 effectiveness
# Is the continuous promotion (Promo2) actually effective?
promo2_summary = df.groupby('Promo2')['Sales'].agg(['mean', 'median', 'count'])
promo2_summary.index = ['No Promo2', 'Promo2']
print("\nPromo2 effectiveness:")
print(promo2_summary)

promo2_lift = (
    promo2_summary.loc['Promo2', 'mean'] -
    promo2_summary.loc['No Promo2', 'mean']
) / promo2_summary.loc['No Promo2', 'mean'] * 100
print(f"\nPromo2 lift: {promo2_lift:.1f}%")

promo1_summary = df.groupby('Promo')['Sales'].agg(['mean', 'median', 'count'])
promo2_summary.index = ['No Promo', 'Promo']
print("\nPromo effectiveness:")
print(promo_summary)

promo_lift = (
    promo_summary.loc['Promo', 'mean'] -
    promo_summary.loc['No Promo', 'mean']
) / promo_summary.loc['No Promo', 'mean'] * 100
print(f"\nPromo lift: {promo_lift:.1f}%")

# Compare Promo vs Promo2 effectiveness
plt.figure()
lifts = pd.Series({
    'Promo (Short-term)': promo_lift,
    'Promo2 (Continuous)': promo2_lift
})
lifts.plot(kind='bar', color=['coral', 'steelblue'])
plt.title('Short-term Promo vs Continuous Promo2 Lift')
plt.ylabel('Sales Lift %')
plt.xticks(rotation=0)
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# %% Cell 10 — Markdown optimization insight
# When should you run promos for maximum impact?
# Combine day of week and month insights

best_day = promo_by_day['Lift_Pct'].idxmax()
worst_day = promo_by_day['Lift_Pct'].idxmin()
best_month = promo_by_month['Lift_Pct'].idxmax()
worst_month = promo_by_month['Lift_Pct'].idxmin()

print("\n--- MARKDOWN OPTIMIZATION INSIGHTS ---")
print(f"Best day to run promotions: Day {best_day} "
      f"(lift: {promo_by_day.loc[best_day, 'Lift_Pct']}%)")
print(f"Worst day to run promotions: Day {worst_day} "
      f"(lift: {promo_by_day.loc[worst_day, 'Lift_Pct']}%)")
print(f"Best month to run promotions: Month {best_month} "
      f"(lift: {promo_by_month.loc[best_month, 'Lift_Pct']}%)")
print(f"Worst month to run promotions: Month {worst_month} "
      f"(lift: {promo_by_month.loc[worst_month, 'Lift_Pct']}%)")
print(f"\nStores with no nearby competition (>10km) vs heavy competition (<500m):")
far_sales = df[df['DistanceBucket'] == '>10km']['Sales'].mean()
close_sales = df[df['DistanceBucket'] == '<500m']['Sales'].mean()
comp_impact = (far_sales - close_sales) / close_sales * 100
print(f"Sales premium for stores with no nearby competition: {comp_impact:.1f}%")

# %% Cell 11 — Summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Retail Pricing & Markdown Analysis Summary', 
             fontsize=14, fontweight='bold')

# Promo lift by store type
promo_by_store['Lift_Pct'].plot(kind='bar', ax=axes[0,0], color='coral')
axes[0,0].set_title('Promo Lift % by Store Type')
axes[0,0].set_xlabel('Store Type')
axes[0,0].set_ylabel('Lift %')
axes[0,0].tick_params(axis='x', rotation=0)

# Promo lift by day
promo_by_day['Lift_Pct'].plot(kind='bar', ax=axes[0,1], color='steelblue')
axes[0,1].set_title('Promo Lift % by Day of Week')
axes[0,1].set_xlabel('Day of Week')
axes[0,1].set_ylabel('Lift %')
axes[0,1].tick_params(axis='x', rotation=0)

# Promo lift by month
promo_by_month['Lift_Pct'].plot(kind='bar', ax=axes[1,0], color='green')
axes[1,0].set_title('Promo Lift % by Month')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('Lift %')
axes[1,0].tick_params(axis='x', rotation=0)

# Competition distance impact
distance_sales.plot(kind='bar', ax=axes[1,1], color='purple')
axes[1,1].set_title('Sales by Competition Distance')
axes[1,1].set_xlabel('Distance to Competitor')
axes[1,1].set_ylabel('Average Sales')
axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

print("\nPricing analysis complete")