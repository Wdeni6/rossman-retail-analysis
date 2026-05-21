# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:19:10 2026

@author: Hunter Denison
"""

# %% Cell 1 — Imports and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

df = pd.read_csv('C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/train_model_ready.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

print("Loaded shape:", df.shape)

# %% Cell 2 — Define features and target
feature_cols = [
    # Calendar features
    'DayOfWeek', 'Month', 'Year', 'WeekOfYear', 'Quarter',
    'IsMonthStart', 'IsMonthEnd',
    # Promotional features
    'Promo', 'Promo2', 'Promo2TenureMonths',
    # Store features
    'StoreTypeEncoded', 'AssortmentEncoded',
    'CompetitionDistance', 'CompetitionTenureMonths', 'IsNewCompetitor',
    # Holiday features
    'SchoolHoliday', 'StateHolidayEncoded',
    # Lag and rolling features
    'Sales_Lag7', 'Sales_Lag14', 'Sales_Lag30',
    'Sales_Rolling7Mean', 'Sales_Rolling30Mean', 'Sales_Rolling7Std'
]

target = 'Sales'

print("Number of features:", len(feature_cols))
print("Target:", target)

# %% Cell 3 — Time-based train/test split
# Training on earlier dates and test on later dates

split_date = '2015-06-01'

train_data = df[df['Date'] < split_date].copy()
test_data = df[df['Date'] >= split_date].copy()

X_train = train_data[feature_cols]
y_train = train_data[target]

X_test = test_data[feature_cols]
y_test = test_data[target]

print(f"\nTraining period: {train_data['Date'].min()} to {train_data['Date'].max()}")
print(f"Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")
print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

# %% Cell 4 — Define evaluation metrics
# RMSPE penalizes percentage errors equally regardless of store size
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe_score = rmspe(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"  MAE:   {mae:,.2f}")
    print(f"  RMSE:  {rmse:,.2f}")
    print(f"  RMSPE: {rmspe_score:.4f} ({rmspe_score*100:.2f}%)")
    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'RMSPE': rmspe_score}

# %% Cell 5 — Baseline model
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_results = evaluate_model("Baseline (Mean)", y_test, baseline_pred)

# %% Cell 6 — Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_results = evaluate_model("Linear Regression", y_test, lr_pred)

# %% Cell 7 — Random Forest
print("\nTraining Random Forest")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    n_jobs=-1,  # Uses all CPU cores
    random_state=2
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_results = evaluate_model("Random Forest", y_test, rf_pred)

# %% Cell 8 — Gradient Boosting (XGBoost via sklearn)
print("\nTraining Gradient Boosting")
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=2
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_results = evaluate_model("Gradient Boosting", y_test, gb_pred)

# %% Cell 9 — Model comparison
results = pd.DataFrame([
    baseline_results, lr_results, rf_results, gb_results
])
results = results.sort_values('RMSPE')

print("\n--- MODEL COMPARISON ---")
print(results.to_string(index=False))

plt.figure()
plt.bar(results['name'], results['RMSPE'], color='steelblue')
plt.title('Model Comparison by RMSPE (lower is better)')
plt.ylabel('RMSPE')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

# %% Cell 10 — Feature importance from best model
best_model = rf  # Change to whichever model wins

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

plt.figure()
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature',
            palette='Blues_r')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# %% Cell 11 — Actual vs predicted plot
plt.figure()
sample = test_data.copy()
sample['Predicted'] = rf_pred

# Plot one store's actual vs predicted sales over time
store_sample = sample[sample['Store'] == 1].sort_values('Date')
plt.plot(store_sample['Date'], store_sample['Sales'],
         label='Actual', alpha=0.7)
plt.plot(store_sample['Date'], store_sample['Predicted'],
         label='Predicted', alpha=0.7)
plt.title('Store 1: Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

# %% Cell 12 — Residual analysis
residuals = y_test - rf_pred

plt.figure()
plt.hist(residuals, bins=50, color='coral')
plt.title('Residual Distribution (Actual - Predicted)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

print("\nResidual stats:")
print(f"Mean residual: {residuals.mean():,.2f}")
print(f"Std residual: {residuals.std():,.2f}")
print(f"% predictions within 10% of actual: "
      f"{(abs(residuals/y_test) < 0.10).mean()*100:.1f}%")
print(f"% predictions within 20% of actual: "
      f"{(abs(residuals/y_test) < 0.20).mean()*100:.1f}%")

# While store one showed strong correlation in the Actual vs Predicted plot there 
# was a pattern of overprediction by 1000 in the peaks.
# Several more stores were checked and this pattern did not persist.
# The residuals plot being normally distributed gives us further validation that errors are not one sided.

# %% Cell 13 — Save predictions
test_data['Predicted_Sales'] = rf_pred
test_data[['Store', 'Date', 'Sales', 'Predicted_Sales']].to_csv(
    'C:/Users/Hunter Denison/Documents/MASTERS/Kaggle/rossmann-retail-analysis/data/predictions.csv', index=False
)
print("\nPredictions saved")