"""
Master Generator for the Complete End-to-End Jupyter / Google Colab Notebook
covering all 9 Steps (Part 1, Part 2, Part 3).
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13"
    }
}

cells = []

# ==============================================================================
# TITLE & OVERVIEW
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Sales Forecasting using Adaptive Basis Function Models
## Complete Machine Learning & Econometric Time Series Project (Steps 1 – 9)

---
### Executive Summary & Problem Formulation
Accurate sales forecasting in multi-echelon retail environments is critical for optimizing working capital, minimizing stockouts, and scheduling staffing. Retail demand time series are characterized by:
1. **Strong deterministic and stochastic seasonality** (52-week annual cycles, Black Friday / Thanksgiving surges, Christmas peaks).
2. **Exogenous macroeconomic influences** (fuel prices, CPI, unemployment, promotional markdowns).
3. **Non-linear interactions** across store types, floor space, and calendar events.

Traditional regression assumes fixed linear relationships $\\hat{y} = \\mathbf{w}^T \\mathbf{x}$. Fixed basis expansions (Polynomial, Fourier) introduce non-linearities but use rigid, data-independent transformations $\\phi_j(\\mathbf{x})$.
This project applies **Adaptive Basis Function Models** where the basis functions $\\phi_j(\\mathbf{x}; \\boldsymbol{\\theta}_j)$ are parameterized and adaptively optimized directly from data:
$$\\hat{y}(\\mathbf{x}) = w_0 + \\sum_{j=1}^M w_j \\phi_j(\\mathbf{x}; \\boldsymbol{\\theta}_j)$$
where both linear weights $w_j$ and basis parameters $\\boldsymbol{\\theta}_j$ (such as hidden layer weights, radial centers, spline knots, or decision tree splits) are learned.

---
### Dataset & Scope
* **Dataset**: Kaggle *Walmart Recruiting - Store Sales Forecasting* (`train.csv`, `features.csv`, `stores.csv`).
* **Timeline**: February 2010 to October 2012 (143 weekly timestamps across 45 stores).
* **Primary Forecasting Scope**: Total weekly sales per store (aggregated across departments) evaluated on a strict 16-week chronological holdout test set with 5-fold expanding-window `TimeSeriesSplit` cross-validation.
"""))

# ==============================================================================
# STEP 0: ENVIRONMENT SETUP
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 0. Environment Setup & Reproducibility
We fix the global random seed to `42` across Python, NumPy, Scikit-Learn, and gradient boosting frameworks. Dynamic path resolution enables execution across Google Colab, JupyterLab, and command-line environments."""))

cells.append(nbf.v4.new_code_cell("""# Environment Configuration & Path Resolution
import os
import sys
import time
import random
import warnings
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# Set random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Formatting
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'Arial'

# Dynamic path resolution
CURRENT_PATH = pathlib.Path('.').resolve()
PROJECT_ROOT = CURRENT_PATH.parent if CURRENT_PATH.name == 'notebooks' else CURRENT_PATH

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, SRC_DIR]:
    os.makedirs(d, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Random seed: {SEED}")
"""))

# ==============================================================================
# STEP 1: DATA UNDERSTANDING & EDA
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Data Understanding & Exploratory Data Analysis (EDA)

We ingest the three raw CSV files, inspect schemas, handle missing promotional markdowns, and aggregate sales to the store level."""))

cells.append(nbf.v4.new_code_cell("""# Ingest Raw Datasets & Verify Panel Balance
stores = pd.read_csv(os.path.join(DATA_DIR, 'stores.csv'))
features = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'))
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

features['Date'] = pd.to_datetime(features['Date'])
train['Date'] = pd.to_datetime(train['Date'])

print(f"stores.csv:   {stores.shape}   | Types: {stores['Type'].value_counts().to_dict()}")
print(f"features.csv: {features.shape} | Date range: {features['Date'].min().date()} to {features['Date'].max().date()}")
print(f"train.csv:    {train.shape} | Date range: {train['Date'].min().date()} to {train['Date'].max().date()}")

# Aggregate sales across departments to store-level
store_weekly = train.groupby(['Store', 'Date', 'IsHoliday'], as_index=False)['Weekly_Sales'].sum()

# Merge master balanced panel
merged = store_weekly.merge(stores, on='Store', how='left')
merged = merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
merged = merged.sort_values(by=['Store', 'Date']).reset_index(drop=True)

# Impute MarkDown NaNs with 0.0 and create indicator
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
merged['markdown_available'] = (
    merged[markdown_cols].notnull().any(axis=1) & 
    (merged[markdown_cols].fillna(0) > 0).any(axis=1)
).astype(int)

for col in markdown_cols:
    merged[col] = merged[col].fillna(0.0)

merged['IsHoliday'] = merged['IsHoliday'].astype(int)

print(f"Master merged store panel: {merged.shape} (Expected: 45 stores * 143 weeks = 6,435 rows)")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 1.2 Diagnostic Time Series Properties: STL Decomposition & ADF Stationarity
We run Seasonal-Trend decomposition using LOESS (STL) with an annual period $P=52$ and execute Augmented Dickey-Fuller (ADF) unit-root tests."""))

cells.append(nbf.v4.new_code_cell("""# STL Decomposition & ADF Stationarity Testing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

chain_weekly = merged.groupby('Date').agg({'Weekly_Sales': 'sum', 'IsHoliday': 'max'}).reset_index()
chain_ts = chain_weekly.set_index('Date')['Weekly_Sales'].asfreq('W-FRI')

stl_res = STL(chain_ts, period=52, robust=True).fit()

# Augmented Dickey-Fuller Tests
adf_raw = adfuller(chain_ts.dropna(), autolag='AIC')
adf_diff = adfuller(chain_ts.diff().dropna(), autolag='AIC')

adf_df = pd.DataFrame({
    'Series': ['Raw Total Sales', 'First Difference (Δ Sales)'],
    'ADF_Statistic': [adf_raw[0], adf_diff[0]],
    'p_value': [adf_raw[1], adf_diff[1]],
    'Critical_Value_5%': [adf_raw[4]['5%'], adf_diff[4]['5%']],
    'Stationary (p < 0.05)': [adf_raw[1] < 0.05, adf_diff[1] < 0.05]
})
display(adf_df)
"""))

# ==============================================================================
# STEP 2: FEATURE ENGINEERING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Zero-Leakage Feature Engineering
We generate 37 features across Calendar/Cyclical, Holiday Proximity, Autoregressive Lags, Shifted Rolling Windows, Store Attributes, and Macroeconomic Indicators. All rolling statistics are strictly calculated with `.shift(1)` to eliminate lookahead bias."""))

cells.append(nbf.v4.new_code_cell("""# Run Feature Pipeline
from src.features import prepare_feature_pipeline, get_feature_columns, time_based_train_test_split, scale_features

df_features = prepare_feature_pipeline(merged, lags=[1, 2, 4, 52], rolling_windows=[4, 12], drop_na_lags=True)
predictor_cols = get_feature_columns(df_features)

print(f"Features engineered shape: {df_features.shape}")
print(f"Predictor feature count:  {len(predictor_cols)}")
"""))

# ==============================================================================
# STEP 3: TRAIN / TEST SPLIT & VALIDATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Train / Test Split & TimeSeries Validation Strategy
We partition the dataset chronologically:
* **Training Partition**: February 2011 to July 6, 2012 (75 weeks $\\times$ 45 stores = 3,375 rows).
* **Holdout Testing Partition**: July 13, 2012 to October 26, 2012 (16 weeks $\\times$ 45 stores = 720 rows).
* `StandardScaler` is fitted **exclusively on the training partition**."""))

cells.append(nbf.v4.new_code_cell("""# Chronological Split and Scaling
train_df, test_df, split_date = time_based_train_test_split(df_features, test_weeks=16)
train_scaled, test_scaled, scaler = scale_features(train_df, test_df, predictor_cols, scaler_type='standard')

X_train = train_scaled[predictor_cols]
y_train = train_scaled['Weekly_Sales'].values
X_test = test_scaled[predictor_cols]
y_test = test_scaled['Weekly_Sales'].values
is_holiday_test = test_df['IsHoliday'].values

print(f"Train split: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({X_train.shape})")
print(f"Test split:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({X_test.shape})")
"""))

# ==============================================================================
# STEP 4: MODELS IMPLEMENTATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Model Architectures & Mathematical Foundations

Every model is conceptualized under the general basis function framework:
$$\\hat{y}(\\mathbf{x}) = w_0 + \\sum_{j=1}^M w_j \\phi_j(\\mathbf{x}; \\boldsymbol{\\theta}_j)$$

### Model Categorization:
1. **Baselines**:
   * **Naive Forecaster**: $\\phi_1(\\mathbf{x}) = y_{t-1}, w_1 = 1$.
   * **Seasonal Naive**: $\\phi_1(\\mathbf{x}) = y_{t-52}, w_1 = 1$.
   * **Linear & Ridge Regression**: $\\phi_j(\\mathbf{x}) = x_j$ with $\\ell_2$ shrinkage.
2. **Fixed Basis Function Models**:
   * **Polynomial Regression**: $\\phi_j(\\mathbf{x}) = x_i^d$ (degree $d=2$). Bases are data-independent.
   * **Fourier Feature Regression**: $\\phi_k(t) = [\\sin(2\\pi k t / P), \\cos(2\\pi k t / P)]$.
3. **Adaptive Basis Function Models**:
   * **Multi-Layer Perceptron (MLP)**: $\\phi_j(\\mathbf{x}) = \\sigma(\\mathbf{w}_j^T \\mathbf{x} + b_j)$. Bases are learned via backpropagation.
   * **Radial Basis Function (RBF) Network**: $\\phi_j(\\mathbf{x}) = \\exp(-\\gamma \\|\\mathbf{x} - \\boldsymbol{\\mu}_j\\|^2)$. Centers $\\boldsymbol{\\mu}_j$ are learned adaptively via $K$-Means clustering.
   * **Multivariate Adaptive Regression Splines (MARS)**: $\\phi_j(\\mathbf{x}) = [\\pm(x_i - t)]_+$. Piecewise linear/cubic splines with adaptive quantile knots.
   * **Generalized Additive Model (GAM)**: $\\hat{y} = \\beta_0 + \\sum f_j(x_j)$ with penalized B-spline bases.
   * **Decision Trees & Random Forests**: $\\phi_m(\\mathbf{x}) = \\mathbb{I}(\\mathbf{x} \\in R_m)$. Adaptive recursive partitioning of the feature space.
   * **Gradient Boosted Trees (GBDT, XGBoost, LightGBM)**: Adaptively adds weak learners via functional gradient descent."""))

# ==============================================================================
# STEP 5: HYPERPARAMETER TUNING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Hyperparameter Tuning via TimeSeriesSplit
We tune tree depth, learning rates, regularization alphas, and hidden layer dimensions using a 3-fold expanding-window `TimeSeriesSplit` on the training partition."""))

cells.append(nbf.v4.new_code_cell("""# Load Saved Tuning Results
tuning_df = pd.read_csv(os.path.join(TABLES_DIR, 'hyperparameter_tuning.csv'))
display(tuning_df)
"""))

# ==============================================================================
# STEP 6: EVALUATION METRICS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Evaluation Metrics on the Holdout Test Set
We evaluate all 15 model variations across MAE, RMSE, MAPE, sMAPE, $R^2$, MASE, and Weighted MAE (WMAE, with $5\\times$ weight on holiday weeks):
$$\\text{WMAE} = \\frac{\\sum_{i=1}^N w_i |y_i - \\hat{y}_i|}{\\sum_{i=1}^N w_i}, \\quad w_i = \\begin{cases} 5 & \\text{if Holiday} \\\\ 1 & \\text{otherwise} \\end{cases}$$"""))

cells.append(nbf.v4.new_code_cell("""# Display Model Comparison Metrics Table
metrics_df = pd.read_csv(os.path.join(TABLES_DIR, 'model_comparison_metrics.csv'))
display(metrics_df[['Model', 'WMAE', 'MAE', 'RMSE', 'MAPE (%)', 'R2', 'MASE', 'Holiday_MAE', 'NonHoliday_MAE']])
"""))

cells.append(nbf.v4.new_code_cell("""# Display Diebold-Mariano Hypothesis Test Result
dm_df = pd.read_csv(os.path.join(TABLES_DIR, 'diebold_mariano_test.csv'))
print("=== DIEBOLD-MARIANO TEST (Best Adaptive vs Best Baseline) ===")
display(dm_df)
"""))

# ==============================================================================
# STEP 7: DIAGNOSTICS & VISUALIZATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 7: Model Diagnostics & Interpretability Analysis
We analyze actual vs predicted trajectories, residual distributions, feature attributions via SHAP, GAM spline partial dependencies, RBF learned centers, learning curves, and feature ablation impacts."""))

cells.append(nbf.v4.new_code_cell("""# Display Ablation Study Results
ablation_df = pd.read_csv(os.path.join(TABLES_DIR, 'ablation_study_results.csv'))
print("=== FEATURE ABLATION STUDY ===")
display(ablation_df)
"""))

cells.append(nbf.v4.new_code_cell("""# Display Forecast Error by Store Type
type_error_df = pd.read_csv(os.path.join(TABLES_DIR, 'error_by_store_type.csv'))
print("=== FORECAST ERROR BY STORE TYPE ===")
display(type_error_df)
"""))

# ==============================================================================
# STEP 8: FINAL FORECAST & BUSINESS INSIGHTS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 8: Multi-Step Ahead Forecasting & Managerial Strategy
The champion model is retrained on all 91 weeks of panel data to generate a **12-week multi-step forward forecast** (November 2012 to January 2013), complete with $80\\%$ and $95\\%$ prediction intervals derived via empirical residual bootstrap."""))

cells.append(nbf.v4.new_code_cell("""# Display 12-Week Forward Forecast Summary
forecast_df = pd.read_csv(os.path.join(TABLES_DIR, 'twelve_week_forecast.csv'))
display(forecast_df)
"""))

# ==============================================================================
# STEP 9: ARTIFACTS VERIFICATION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Step 9: Project Artifacts & File Deliverables Audit
All generated publication-grade figures, metric tables, and academic report drafts are verified below."""))

cells.append(nbf.v4.new_code_cell("""# Audit Output Artifacts
print("=== GENERATED FIGURES (results/figures/) ===")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
        print(f"  - {f:35s} ({size_kb:,.1f} KB)")

print("\\n=== GENERATED METRIC TABLES (results/tables/) ===")
for t in sorted(os.listdir(TABLES_DIR)):
    if t.endswith('.csv'):
        size_b = os.path.getsize(os.path.join(TABLES_DIR, t))
        print(f"  - {t:35s} ({size_b:,} bytes)")
"""))

nb.cells = cells

# Save notebook
notebook_path = 'notebooks/sales_forecasting_adaptive_basis_models.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Master Notebook successfully created at: {notebook_path}")
print(f"Total cells: {len(cells)}")
