# Rossmann Retail Sales Intelligence
### Forecasting, Pricing Analysis & Markdown Optimization

---

## Overview

This project applies end-to-end data science to the Rossmann Store Sales dataset with 
over 800,000 daily sales records across 1,000+ German drugstore locations. The goal 
is to answer three practical business questions:

1. **What will sales be next month?** (demand forecasting)
2. **How much do promotions actually lift sales and when do they work best?** 
   (price elasticity & promo analysis)
3. **Is the continuous promotion program worth running?** (markdown optimization)

The analysis moves from raw data through cleaning, feature engineering, predictive 
modeling, and actionable business recommendations.

---

## Key Findings

| Finding | Result |
|---|---|
| Short-term promo lift | 38.7% average sales increase |
| Best day to run promotions | Monday (highest lift) |
| Worst day to run promotions | Friday (lowest lift) |
| Promo lift varies by store type | Range: 17.9% to 42.9% |
| Continuous Promo2 effectiveness | -10.7% (negative lift) |
| Forecast accuracy within 10% | 57.5% of predictions |
| Forecast accuracy within 20% | 85.8% of predictions |
| Strongest sales predictor | 30-day rolling average sales |

---

## Business Recommendations

**1. Prioritize Monday and early-week promotions**
Promotional lift declines consistently from Monday through Friday. Early-week 
shoppers are more intentional and price-responsive. Reallocating promotional spend 
toward Monday–Wednesday could improve ROI without increasing total spend.

**2. Allocate promotional budget by store type**
Promo lift ranges from 17.9% to 42.9% across store types. Promotional investment 
is not equally efficient across all formats. The promo usage should be weighted toward 
highest-responding store types.

**3. Reconsider the Promo2 continuous promotion program**
Stores participating in Promo2 show a -10.7% sales correlation vs non-participating 
stores, contrasted with +33-40% lift from short-term promotions. This pattern is 
consistent with promotion fatigue. Customers stop perceiving value when discounts 
are always available. Replacing Promo2 with periodic short-term events 
would likely restore perception of customer's opportunity and improve lift.

**4. Use recent sales momentum as the primary forecasting signal**
The 30-day rolling sales average was the single most predictive feature — more 
predictive than promotions or day of week. Store-level momentum should anchor any 
inventory planning or staffing model.

**5. Amplify December promotions, manage January/February expectations**
December promotional lift reaches the low 40s%, amplifying already strong holiday 
traffic. January and February show lower baseline sales with reduced promotional 
responsiveness. 

---

## Project Structure

rossmann-retail-analysis/
│
├── data/                          # Data files (not included in repo)
│   ├── train.csv                  # Raw training data
│   ├── store.csv                  # Store metadata
│   ├── train_cleaned.csv          # After cleaning
│   ├── train_featured.csv         # After feature engineering
│   ├── train_model_ready.csv      # Final model input
│   └── predictions.csv            # Model output
│
├── notebooks/
│   ├── 01_eda_and_cleaning.py     # Exploratory data analysis
│   ├── 02_data_cleaning.py        # Data cleaning pipeline
│   ├── 03_feature_engineering.py  # Feature construction
│   ├── 04_modeling.py             # Forecasting model
│   └── 05_pricing_analysis.py     # Pricing & markdown analysis
│
├── README.md
└── README.pdf

---

## Data

Dataset: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)  
Source: Kaggle — Dirk Rossmann GmbH  
To reproduce this analysis, download `train.csv` and `store.csv` from the link 
above and place them in the `/data` folder.

**Dataset overview:**
- 1,017,209 raw records across 1,115 stores
- Date range: January 2013 – July 2015
- 18 features after merging store metadata
- 844,338 records after removing closed store days and zero sale days
- 810,888 records after feature engineering and dropping small amount of lag feature nulls

---

## Methodology

### Data Cleaning
- Removed closed store days (no predictive value for sales modeling)
- Identified and removed 54 anomalous records: stores marked open with zero 
  sales and zero customers.They no consistent pattern across store type, 
  day of week, or promotional status. They were treated as data entry errors.
- Imputed store file missing values with business-reasoned defaults 
  (e.g. missing CompetitionDistance filled with 2x maximum observed distance 
  to represent no known nearby competitor so we could keep these in the model.)

### Feature Engineering
38 total fields after constructing features from raw data including:
- **Calendar features:** year, month, day, week of year, quarter, 
  month-start/end flags
- **Promotional features:** Promo2 participation tenure
- **Competition features:** distance buckets, competition tenure in months, 
  new competitor flag
- **Time series features:** 7/14/30-day sales lags, 7/30-day rolling means, 
  7-day rolling standard deviation
- **Business metric:** sales per customer (revenue efficiency)

### Modeling
Four models evaluated using Root Mean Squared Percentage Error (RMSPE):
- Baseline (mean prediction)
- Linear Regression
- Random Forest
- Gradient Boosting

Time-based train/test split used throughout. Training was done on data before 
June 2015 and testing on June–July 2015 to simulate real forecasting conditions 
and prevent data leakage.

**Random Forest achieved the best performance** with RMSPE of 0.16 and 57.5% of predictions 
falling within 10% of actual sales. The 30-day rolling sales average was the 
single most important feature, confirming that recent store momentum is the 
strongest signal for near-term forecasting.

### Pricing Analysis
Promotional effectiveness analyzed across store type, day of week, month, 
and competition context. Statistical significance confirmed via t-test 
(p < 0.0001). Key contrast identified between short-term promotional lift 
(+38.7%) and continuous Promo2 program (-10.7%), suggesting promotion 
fatigue as a meaningful factor in this retail context.

---

## Technical Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | Modeling |
| matplotlib | Visualization |
| seaborn | Statistical visualization |
| SciPy | Statistical testing |

---

## About

Built as a portfolio project to demonstrate end-to-end data science skills 
including data cleaning, feature engineering, time series forecasting, and 
retail pricing analysis.

*Data sourced from Kaggle Rossmann Store Sales competition. Raw data files 
are not included in this repository per competition data use guidelines.*
