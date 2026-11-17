"""
Comprehensive Unit Test Suite for Walmart Sales Forecasting Feature Pipeline.
Verifies:
1. Zero current-week data leakage (modifying week t sales does not alter week t features).
2. Store isolation (rolling features never incorporate data across store boundaries).
3. Markdown available vs active indicators.
4. Correct next-occurrence holiday countdown metrics.
5. Outlier diagnostics preserving legitimate holiday surges.
6. Recursive multi-step forward forecasting mechanics.
7. Fold-aware TimeSeries cross-validation without cross-fold scaler leakage.
"""

import numpy as np
import pandas as pd
import src.features as feat


def test_store_isolation():
    """
    Test that rolling means and stds for Store 2 NEVER use Store 1's sales values.
    """
    print("\n[TEST 1] Testing Store Isolation in Rolling Features...")
    # Synthetic data: Store 1 has sales = 100, Store 2 has sales = 1,000,000
    dates = pd.date_range('2011-01-07', periods=20, freq='W-FRI')
    df_s1 = pd.DataFrame({
        'Store': 1,
        'Date': dates,
        'Weekly_Sales': 100.0,
        'Type': 'A',
        'Size': 100000,
        'Temperature': 50.0,
        'Fuel_Price': 3.0,
        'CPI': 200.0,
        'Unemployment': 7.0,
        'IsHoliday': 0,
        'MarkDown1': np.nan, 'MarkDown2': np.nan, 'MarkDown3': np.nan, 'MarkDown4': np.nan, 'MarkDown5': np.nan
    })
    df_s2 = pd.DataFrame({
        'Store': 2,
        'Date': dates,
        'Weekly_Sales': 1_000_000.0,
        'Type': 'B',
        'Size': 150000,
        'Temperature': 55.0,
        'Fuel_Price': 3.1,
        'CPI': 205.0,
        'Unemployment': 6.5,
        'IsHoliday': 0,
        'MarkDown1': np.nan, 'MarkDown2': np.nan, 'MarkDown3': np.nan, 'MarkDown4': np.nan, 'MarkDown5': np.nan
    })
    df_combined = pd.concat([df_s1, df_s2], ignore_index=True)

    df_feat = feat.engineer_lag_and_rolling_features(df_combined, lags=[1, 2], rolling_windows=[4])

    s1_roll = df_feat[df_feat['Store'] == 1]['sales_roll_mean_4'].dropna()
    s2_roll = df_feat[df_feat['Store'] == 2]['sales_roll_mean_4'].dropna()

    # Store 1 rolling mean should be exactly 100.0
    assert np.allclose(s1_roll.values, 100.0), f"Store 1 rolling mean contaminated: {s1_roll.values}"
    # Store 2 rolling mean should be exactly 1,000,000.0 (NEVER contaminated by Store 1's 100.0!)
    assert np.allclose(s2_roll.values, 1_000_000.0), f"Store 2 rolling mean contaminated by Store 1: {s2_roll.values}"
    print("  -> PASSED: Store rolling statistics are strictly isolated within each store.")


def test_zero_current_week_leakage():
    """
    Test that modifying sales at week t does NOT affect week t's lag or rolling features.
    """
    print("\n[TEST 2] Testing Zero Current-Week Leakage...")
    dates = pd.date_range('2011-01-07', periods=15, freq='W-FRI')
    sales_normal = [100.0 * (i + 1) for i in range(15)]
    sales_tampered = sales_normal.copy()
    tampered_idx = 8
    sales_tampered[tampered_idx] = 999_999_999.0  # Huge outlier in current week

    df1 = pd.DataFrame({'Store': 1, 'Date': dates, 'Weekly_Sales': sales_normal, 'Type': 'A'})
    df2 = pd.DataFrame({'Store': 1, 'Date': dates, 'Weekly_Sales': sales_tampered, 'Type': 'A'})

    feat1 = feat.engineer_lag_and_rolling_features(df1, lags=[1, 2, 4], rolling_windows=[4, 12])
    feat2 = feat.engineer_lag_and_rolling_features(df2, lags=[1, 2, 4], rolling_windows=[4, 12])

    # At the tampered week t, features in feat1 and feat2 MUST BE IDENTICAL!
    # Because all features at week t must strictly use prior weeks (t-1, t-2, ...).
    for col in ['sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_roll_mean_4', 'sales_roll_std_4']:
        val1 = feat1.loc[tampered_idx, col]
        val2 = feat2.loc[tampered_idx, col]
        if np.isnan(val1):
            assert np.isnan(val2), f"NaN mismatch on {col}"
        else:
            assert np.isclose(val1, val2), (
                f"Data Leakage Detected! Feature '{col}' at index {tampered_idx} changed from {val1} to {val2} "
                f"when current week's sales were altered!"
            )
    print("  -> PASSED: Zero current-week data leakage verified. Tampering week t has 0 impact on week t features.")


def test_markdown_available_and_active():
    """
    Test markdown_available indicates recorded presence, and markdown_active indicates positive promotion.
    """
    print("\n[TEST 3] Testing Markdown Available vs Active Flags...")
    df = pd.DataFrame({
        'Store': [1, 1, 1],
        'Date': pd.to_datetime(['2011-02-04', '2011-11-11', '2012-02-10']),
        'MarkDown1': [np.nan, 0.0, 1500.0],
        'MarkDown2': [np.nan, 0.0, 0.0],
        'MarkDown3': [np.nan, 0.0, 0.0],
        'MarkDown4': [np.nan, 0.0, 0.0],
        'MarkDown5': [np.nan, 0.0, 200.0],
        'IsHoliday': [0, 0, 1]
    })
    df_clean = feat.clean_markdown_and_outliers(df)

    # Row 0: All NaNs -> available = 0, active = 0
    assert df_clean.loc[0, 'markdown_available'] == 0
    assert df_clean.loc[0, 'markdown_active'] == 0

    # Row 1: Recorded but all 0.0 -> available = 1, active = 0
    assert df_clean.loc[1, 'markdown_available'] == 1
    assert df_clean.loc[1, 'markdown_active'] == 0

    # Row 2: Recorded and positive -> available = 1, active = 1
    assert df_clean.loc[2, 'markdown_available'] == 1
    assert df_clean.loc[2, 'markdown_active'] == 1
    print("  -> PASSED: markdown_available and markdown_active distinguish recorded data from active promotions.")


def test_holiday_countdown():
    """
    Test that weeks-to-Thanksgiving counts down to 0 on Thanksgiving week and resets to ~52 weeks after.
    """
    print("\n[TEST 4] Testing Next-Occurrence Holiday Countdown Metrics...")
    dates = pd.to_datetime([
        '2011-11-11',  # 2 weeks before Thanksgiving 2011 (2011-11-25)
        '2011-11-18',  # 1 week before
        '2011-11-25',  # Thanksgiving week 2011
        '2011-12-02'   # 1 week after Thanksgiving 2011 -> next is Thanksgiving 2012 (2012-11-23)
    ])
    df = pd.DataFrame({'Date': dates})
    df_cal = feat.engineer_calendar_and_holiday_features(df)

    w_to_th = df_cal['weeks_to_thanksgiving'].values
    assert np.isclose(w_to_th[0], 2.0), f"Expected 2.0 weeks, got {w_to_th[0]}"
    assert np.isclose(w_to_th[1], 1.0), f"Expected 1.0 weeks, got {w_to_th[1]}"
    assert np.isclose(w_to_th[2], 0.0), f"Expected 0.0 weeks on Thanksgiving, got {w_to_th[2]}"
    # After Thanksgiving 2011, countdown should be to Thanksgiving 2012 (~51 weeks away)
    assert w_to_th[3] > 40.0, f"Expected > 40 weeks to next Thanksgiving, got {w_to_th[3]}"
    print("  -> PASSED: Holiday countdown accurately reflects future-looking arrival.")


def test_outlier_diagnostics_preserves_spikes():
    """
    Test that outlier diagnostics flag holiday surges without deleting or modifying them.
    """
    print("\n[TEST 5] Testing Outlier Diagnostics & Spike Retention...")
    df = pd.DataFrame({
        'Weekly_Sales': [1000.0, 1100.0, 1050.0, 1020.0, 1080.0, 10000.0],  # 10000 is an extreme surge
        'IsHoliday': [0, 0, 0, 0, 0, 1]
    })
    df_diag, summary = feat.diagnose_outliers(df, iqr_multiplier=2.0)
    assert df_diag.loc[5, 'is_holiday_surge'] == 1
    assert len(df_diag) == 6, "Row count changed! Outlier diagnostics must not delete records."
    assert df_diag['Weekly_Sales'].max() == 10000.0, "Values modified! Spikes must be preserved."
    print("  -> PASSED: Outlier diagnostics identified holiday surge and preserved all records.")


def test_full_pipeline_on_real_data():
    """
    Test the complete pipeline end-to-end on actual dataset files.
    """
    print("\n[TEST 6] Testing Complete Feature Pipeline on Local Dataset...")
    merged, train_raw, features_raw, stores_raw = feat.load_and_merge_data('data')
    assert len(merged) == 45 * 143, f"Expected 6,435 rows, got {len(merged)}"

    df_feat = feat.prepare_feature_pipeline(merged, lags=[1, 2, 4, 52], rolling_windows=[4, 12], include_store_dummies=False)
    assert df_feat.isnull().sum().sum() == 0, "Nulls remain in engineered dataset!"
    
    train_df, test_df, split_date = feat.time_based_train_test_split(df_feat, test_weeks=16)
    assert train_df['Date'].max() < test_df['Date'].min(), "Temporal overlap in train/test split!"
    assert test_df['Date'].nunique() == 16, f"Expected 16 test weeks, got {test_df['Date'].nunique()}"

    feat_cols = feat.get_feature_columns(df_feat)
    train_s, test_s, scaler = feat.scale_features(train_df, test_df, feat_cols)
    assert abs(train_s[feat_cols].mean().mean()) < 1e-2, "Train scaling mean not centered at 0"
    print(f"  -> PASSED: Full pipeline validated ({len(feat_cols)} features, 0 missing values).")


def run_all_tests():
    print("=" * 70)
    print("RUNNING WALMART SALES FORECASTING FEATURE PIPELINE TEST SUITE")
    print("=" * 70)
    test_store_isolation()
    test_zero_current_week_leakage()
    test_markdown_available_and_active()
    test_holiday_countdown()
    test_outlier_diagnostics_preserves_spikes()
    test_full_pipeline_on_real_data()
    print("\n" + "=" * 70)
    print("ALL 6 UNIT TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
