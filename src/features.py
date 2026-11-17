"""
Feature Engineering and Data Preparation Module for Walmart Sales Forecasting.
Adheres strictly to zero data leakage standards (expanding/rolling shifted windows,
groupwise store isolation, time-based splitting, and train-only fitted scalers).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit


def load_raw_data(data_dir='./data'):
    """
    Loads raw stores.csv, features.csv, and train.csv from local directory or Kaggle fallback.
    """
    stores_path = os.path.join(data_dir, 'stores.csv')
    features_path = os.path.join(data_dir, 'features.csv')
    train_path = os.path.join(data_dir, 'train.csv')

    if not (os.path.exists(stores_path) and os.path.exists(features_path) and os.path.exists(train_path)):
        raise FileNotFoundError(
            f"Missing required CSV files in {data_dir}. "
            "Please ensure stores.csv, features.csv, and train.csv are present."
        )

    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)
    train = pd.read_csv(train_path)

    # Standardize Date types
    features['Date'] = pd.to_datetime(features['Date'])
    train['Date'] = pd.to_datetime(train['Date'])

    return stores, features, train


def load_and_merge_data(data_dir='./data', aggregate_store_level=True):
    """
    Loads raw data and merges on Store + Date (+ IsHoliday).
    If aggregate_store_level is True, sums Weekly_Sales across departments per store-week.
    
    Validates that:
    1. Row counts are strictly preserved.
    2. No silent NaNs are produced in core exogenous features (Temperature, Fuel_Price, CPI, Unemployment, Size, Type).
    """
    stores, features, train = load_raw_data(data_dir)

    if aggregate_store_level:
        # Aggregate weekly sales per store across departments
        store_weekly = train.groupby(['Store', 'Date', 'IsHoliday'], as_index=False)['Weekly_Sales'].sum()
        merged = store_weekly.merge(stores, on='Store', how='left')
        merged = merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        expected_len = len(store_weekly)
    else:
        # Department level
        merged = train.merge(stores, on='Store', how='left')
        merged = merged.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        expected_len = len(train)

    # Verification checks
    if len(merged) != expected_len:
        raise ValueError(f"Merge error: expected {expected_len} rows, but got {len(merged)}.")

    core_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Type']
    missing_core = merged[core_cols].isnull().sum()
    if missing_core.any():
        raise ValueError(f"Silent merge failure: missing values detected in core features:\n{missing_core[missing_core > 0]}")

    merged = merged.sort_values(by=['Store', 'Date']).reset_index(drop=True)
    return merged, train, features, stores


def clean_markdown_and_outliers(df):
    """
    Handles missing values in MarkDown1-5:
    - 'markdown_available': 1 if markdown information was recorded (not all NaNs), 0 if untracked.
    - 'markdown_active': 1 if any recorded markdown is strictly positive (> 0.0), 0 otherwise.
    - Fills NaNs in MarkDown1-5 with 0.0.
    - Formats IsHoliday to integer (0 or 1).
    """
    df = df.copy()
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    # 1. markdown_available: indicates whether markdown data was recorded for that store-week
    df['markdown_available'] = df[markdown_cols].notnull().any(axis=1).astype(int)

    # 2. markdown_active: indicates whether any promotional markdown was actively non-zero
    df['markdown_active'] = (df[markdown_cols].fillna(0.0) > 0.0).any(axis=1).astype(int)

    # 3. Fill NaNs with 0.0
    for col in markdown_cols:
        df[col] = df[col].fillna(0.0)

    # Convert IsHoliday to int
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    return df


def diagnose_outliers(df, sales_col='Weekly_Sales', holiday_col='IsHoliday', iqr_multiplier=3.0):
    """
    Diagnostic outlier analysis that identifies extreme values without deleting legitimate holiday sales spikes.
    
    Distinguishes:
    - Holiday sales surges: extreme demand spikes occurring during verified retail holiday events (Thanksgiving, Christmas).
      These are legitimate economic phenomena that models must learn to capture.
    - Non-holiday anomalies: statistical anomalies during standard trading weeks.
    
    Returns diagnostic dataframe and summary statistics dictionary.
    """
    df = df.copy()
    q1 = df[sales_col].quantile(0.25)
    q3 = df[sales_col].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_multiplier * iqr
    lower_bound = max(0.0, q1 - iqr_multiplier * iqr)

    df['is_statistical_outlier'] = ((df[sales_col] > upper_bound) | (df[sales_col] < lower_bound)).astype(int)
    df['is_holiday_surge'] = ((df['is_statistical_outlier'] == 1) & (df[holiday_col] == 1)).astype(int)
    df['is_non_holiday_anomaly'] = ((df['is_statistical_outlier'] == 1) & (df[holiday_col] == 0)).astype(int)

    summary = {
        'total_rows': len(df),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'upper_bound': upper_bound,
        'total_statistical_outliers': df['is_statistical_outlier'].sum(),
        'legitimate_holiday_surges': df['is_holiday_surge'].sum(),
        'non_holiday_anomalies': df['is_non_holiday_anomaly'].sum(),
        'action_taken': 'Preserved all records intact; holiday surges retained for nonlinear adaptive learning.'
    }
    return df, summary


def get_actual_holiday_dates():
    """
    Returns actual calendar dates for Walmart Thanksgiving and Christmas periods across the timeline (2010 to 2013).
    Walmart dataset dates represent Friday weekly ends.
    """
    thanksgiving_dates = pd.to_datetime([
        '2010-11-26',  # Friday after Thanksgiving 2010 (Week 47)
        '2011-11-25',  # Friday after Thanksgiving 2011 (Week 47)
        '2012-11-23',  # Friday after Thanksgiving 2012 (Week 47)
        '2013-11-29'   # Friday after Thanksgiving 2013 (Week 48)
    ])
    christmas_dates = pd.to_datetime([
        '2010-12-31',  # Walmart holiday week for Christmas 2010 (Week 52)
        '2011-12-30',  # Walmart holiday week for Christmas 2011 (Week 52)
        '2012-12-28',  # Walmart holiday week for Christmas 2012 (Week 52)
        '2013-12-27'   # Walmart holiday week for Christmas 2013 (Week 52)
    ])
    return thanksgiving_dates, christmas_dates


def engineer_calendar_and_holiday_features(df):
    """
    Builds temporal calendar features, sin/cos cyclical encodings,
    and forward-looking weeks-to-holiday countdowns using the next annual occurrence of Thanksgiving and Christmas.
    """
    df = df.copy()
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df['Date'] = pd.to_datetime(df['Date'])

    # Calendar features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter

    # Cyclical encodings (preserve cyclic continuity between week 52 and week 1)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.1775)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.1775)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Official Walmart competition holiday weeks:
    # 1. Super Bowl: Week 6
    # 2. Labor Day: Week 36
    # 3. Thanksgiving: Week 47
    # 4. Christmas: Week 51 (or 52)
    df['is_super_bowl'] = (df['week_of_year'] == 6).astype(int)
    df['is_labor_day'] = (df['week_of_year'] == 36).astype(int)
    df['is_thanksgiving'] = (df['week_of_year'] == 47).astype(int)
    df['is_christmas'] = (df['week_of_year'].isin([51, 52])).astype(int)

    # Proximity features: Forward countdown to NEXT occurrence of Thanksgiving and Christmas
    th_dates, ch_dates = get_actual_holiday_dates()

    def calc_weeks_to_next(current_date, holiday_dates):
        future_hols = holiday_dates[holiday_dates >= current_date]
        if len(future_hols) > 0:
            next_hol = future_hols[0]
            days_diff = (next_hol - current_date).days
            return float(days_diff) / 7.0
        else:
            # Fallback if beyond last recorded date: approx 52 weeks minus elapsed
            return 52.0

    df['weeks_to_thanksgiving'] = df['Date'].apply(lambda d: calc_weeks_to_next(d, th_dates))
    df['weeks_to_christmas'] = df['Date'].apply(lambda d: calc_weeks_to_next(d, ch_dates))

    return df


def engineer_lag_and_rolling_features(df, lags=[1, 2, 4, 52], rolling_windows=[4, 12], include_store_dummies=False):
    """
    Generates historical lags and rolling statistics per store.
    
    CRITICAL LEAKAGE-FREE ARCHITECTURE:
    1. Calculations are executed strictly WITHIN each Store using groupwise transform.
    2. Rolling windows use s.shift(1).rolling(w) to guarantee that Store S never sees Store S-1,
       and week t never sees week t's sales.
    """
    df = df.copy()
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)

    # Historical lags within each store
    for l in lags:
        df[f'sales_lag_{l}'] = df.groupby('Store')['Weekly_Sales'].shift(l)

    # Rolling statistics strictly groupwise and shifted by 1 week
    for w in rolling_windows:
        # Rolling Mean: shift(1) ensures week t is excluded
        df[f'sales_roll_mean_{w}'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda s: s.shift(1).rolling(window=w).mean()
        )
        # Rolling Std: shift(1) ensures week t is excluded
        df[f'sales_roll_std_{w}'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda s: s.shift(1).rolling(window=w).std()
        )

    # One-hot encode Store Type
    type_dummies = pd.get_dummies(df['Type'], prefix='Type', dtype=int)
    df = pd.concat([df, type_dummies], axis=1)

    # Optional: One-hot encode Store ID for pooled models to learn store-specific fixed effects
    if include_store_dummies:
        store_dummies = pd.get_dummies(df['Store'], prefix='Store_ID', dtype=int)
        df = pd.concat([df, store_dummies], axis=1)

    return df


def prepare_feature_pipeline(df, lags=[1, 2, 4, 52], rolling_windows=[4, 12], include_store_dummies=False, drop_na_lags=True):
    """
    Runs full feature engineering pipeline sequentially.
    """
    df_clean = clean_markdown_and_outliers(df)
    df_cal = engineer_calendar_and_holiday_features(df_clean)
    df_feat = engineer_lag_and_rolling_features(
        df_cal, lags=lags, rolling_windows=rolling_windows, include_store_dummies=include_store_dummies
    )

    if drop_na_lags:
        # Drop initial burn-in period where 52-week lag is NaN
        df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat


def time_based_train_test_split(df, test_weeks=16):
    """
    Splits the dataset strictly chronologically.
    Last `test_weeks` are assigned to holdout test, preceding weeks to train.
    """
    unique_dates = sorted(df['Date'].unique())
    split_date = unique_dates[-test_weeks]

    train_df = df[df['Date'] < split_date].copy().reset_index(drop=True)
    test_df = df[df['Date'] >= split_date].copy().reset_index(drop=True)

    return train_df, test_df, split_date


def get_feature_columns(df):
    """
    Returns list of predictor feature columns, separating meta columns (Store, Date, Type).
    """
    exclude_cols = ['Store', 'Date', 'Type', 'Weekly_Sales', 'is_statistical_outlier', 'is_holiday_surge', 'is_non_holiday_anomaly']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def scale_features(X_train, X_test, feature_cols, scaler_type='standard'):
    """
    Fits scaler STRICTLY on training set, then transforms both train and test.
    Returns scaled copies of X_train and X_test along with fitted scaler.
    """
    scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# FOLD-AWARE TIME SERIES CROSS-VALIDATION TRANSFORMER (ZERO LEAKAGE ACROSS FOLDS)
# ==============================================================================
class TimeSeriesFoldTransformer:
    """
    Generator that performs TimeSeriesSplit cross-validation while fitting
    the feature scaler STRICTLY on each fold's training index, eliminating cross-fold leakage.
    """
    def __init__(self, n_splits=5, scaler_type='standard'):
        self.n_splits = n_splits
        self.scaler_type = scaler_type
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def split(self, df, feature_cols, target_col='Weekly_Sales'):
        unique_dates = np.array(sorted(df['Date'].unique()))
        
        for fold, (train_d_idx, val_d_idx) in enumerate(self.tscv.split(unique_dates)):
            train_dates = unique_dates[train_d_idx]
            val_dates = unique_dates[val_d_idx]

            train_fold = df[df['Date'].isin(train_dates)].copy()
            val_fold = df[df['Date'].isin(val_dates)].copy()

            # Fit scaler strictly on this fold's training data
            scaler = StandardScaler() if self.scaler_type == 'standard' else RobustScaler()
            train_fold[feature_cols] = scaler.fit_transform(train_fold[feature_cols])
            val_fold[feature_cols] = scaler.transform(val_fold[feature_cols])

            X_tr = train_fold[feature_cols]
            y_tr = train_fold[target_col].values
            X_va = val_fold[feature_cols]
            y_va = val_fold[target_col].values

            yield fold, X_tr, y_tr, X_va, y_va, scaler


# ==============================================================================
# LEAKAGE-FREE RECURSIVE MULTI-STEP FORWARD FORECASTER
# ==============================================================================
def generate_recursive_multi_step_forecast(model, historical_df, future_dates, feature_cols, scaler, horizon=12):
    """
    Executes a leakage-free recursive multi-step forecasting strategy for 12 weeks ahead.
    
    Methodology:
    At each future step h (1 to horizon):
    1. Calendar and holiday features are derived deterministically for future date d_h.
    2. Exogenous macro covariates (Temperature, Fuel_Price, CPI, Unemployment, MarkDowns)
       are drawn from the known features schedule or carried forward.
    3. Endogenous lag features (Lag 1, 2, 4, 52) and rolling statistics (Mean 4, Std 4, Mean 12, Std 12)
       are computed dynamically from a rolling buffer of actual historical sales + previously generated predictions.
    4. Features are standardized using the pre-fitted scaler.
    5. The model predicts sales for all stores at step h.
    6. Predictions are appended to each store's buffer and become the lag inputs for step h+1.
    """
    # Group history by store
    store_buffers = {}
    last_store_rows = {}
    for s_id in sorted(historical_df['Store'].unique()):
        s_data = historical_df[historical_df['Store'] == s_id].sort_values('Date')
        store_buffers[s_id] = list(s_data['Weekly_Sales'].values)
        last_store_rows[s_id] = s_data.iloc[-1].to_dict()

    th_dates, ch_dates = get_actual_holiday_dates()
    forecast_records = []

    for h_idx, f_date in enumerate(future_dates[:horizon], start=1):
        w_of_yr = f_date.isocalendar().week
        is_hol = 1 if w_of_yr in [6, 36, 47, 51, 52] else 0

        # Build feature rows for all stores at this time step
        step_rows = []
        for s_id in sorted(historical_df['Store'].unique()):
            buf = store_buffers[s_id]
            last_r = last_store_rows[s_id]

            # Next holiday countdowns
            w_to_th = (th_dates[th_dates >= f_date][0] - f_date).days / 7.0 if len(th_dates[th_dates >= f_date]) > 0 else 52.0
            w_to_ch = (ch_dates[ch_dates >= f_date][0] - f_date).days / 7.0 if len(ch_dates[ch_dates >= f_date]) > 0 else 52.0

            # Lags dynamically from buffer
            l1 = buf[-1]
            l2 = buf[-2] if len(buf) >= 2 else l1
            l4 = buf[-4] if len(buf) >= 4 else l2
            l52 = buf[-52] if len(buf) >= 52 else buf[0]

            # Rolling stats dynamically from buffer
            r4_mean = np.mean(buf[-4:])
            r4_std = np.std(buf[-4:], ddof=1) if len(buf) >= 4 else 0.0
            r12_mean = np.mean(buf[-12:])
            r12_std = np.std(buf[-12:], ddof=1) if len(buf) >= 12 else 0.0

            row = {
                'Store': s_id,
                'Date': f_date,
                'IsHoliday': is_hol,
                'Size': last_r['Size'],
                'Type': last_r['Type'],
                'Temperature': last_r['Temperature'],
                'Fuel_Price': last_r['Fuel_Price'],
                'CPI': last_r['CPI'],
                'Unemployment': last_r['Unemployment'],
                'MarkDown1': last_r['MarkDown1'],
                'MarkDown2': last_r['MarkDown2'],
                'MarkDown3': last_r['MarkDown3'],
                'MarkDown4': last_r['MarkDown4'],
                'MarkDown5': last_r['MarkDown5'],
                'markdown_available': 1,
                'markdown_active': 1 if any(last_r.get(f'MarkDown{i}', 0) > 0 for i in range(1, 6)) else 0,
                'year': f_date.year,
                'month': f_date.month,
                'week_of_year': w_of_yr,
                'quarter': f_date.quarter,
                'week_sin': np.sin(2 * np.pi * w_of_yr / 52.1775),
                'week_cos': np.cos(2 * np.pi * w_of_yr / 52.1775),
                'month_sin': np.sin(2 * np.pi * f_date.month / 12.0),
                'month_cos': np.cos(2 * np.pi * f_date.month / 12.0),
                'is_super_bowl': 1 if w_of_yr == 6 else 0,
                'is_labor_day': 1 if w_of_yr == 36 else 0,
                'is_thanksgiving': 1 if w_of_yr == 47 else 0,
                'is_christmas': 1 if w_of_yr in [51, 52] else 0,
                'weeks_to_thanksgiving': w_to_th,
                'weeks_to_christmas': w_to_ch,
                'sales_lag_1': l1,
                'sales_lag_2': l2,
                'sales_lag_4': l4,
                'sales_lag_52': l52,
                'sales_roll_mean_4': r4_mean,
                'sales_roll_std_4': r4_std,
                'sales_roll_mean_12': r12_mean,
                'sales_roll_std_12': r12_std,
                'Type_A': 1 if last_r['Type'] == 'A' else 0,
                'Type_B': 1 if last_r['Type'] == 'B' else 0,
                'Type_C': 1 if last_r['Type'] == 'C' else 0,
            }
            # Optional store dummies if present in feature_cols
            for c in feature_cols:
                if c.startswith('Store_ID_'):
                    s_target = int(c.replace('Store_ID_', ''))
                    row[c] = 1 if s_id == s_target else 0

            step_rows.append(row)

        step_df = pd.DataFrame(step_rows)
        # Transform features
        step_scaled = step_df.copy()
        step_scaled[feature_cols] = scaler.transform(step_df[feature_cols])

        # Predict step h sales
        preds_h = model.predict(step_scaled[feature_cols])
        step_df['Forecast_Sales'] = preds_h

        # Update each store's buffer with its own prediction for recursive input in step h+1
        for s_idx, s_id in enumerate(sorted(historical_df['Store'].unique())):
            store_buffers[s_id].append(float(preds_h[s_idx]))

        forecast_records.append(step_df)

    forecast_all = pd.concat(forecast_records, axis=0).reset_index(drop=True)
    return forecast_all
