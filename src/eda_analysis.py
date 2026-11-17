"""
Script to execute complete EDA, STL decomposition, ADF stationarity testing,
and TimeSeriesSplit cross-validation visualization for Steps 1-3.
Saves all figures to results/figures/ and tables to results/tables/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit

import src.features as feat

# Set plotting style and random seed
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
np.random.seed(42)

FIGURES_DIR = 'results/figures'
TABLES_DIR = 'results/tables'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def run_eda_and_generate_artifacts():
    print("Loading raw and merged data...")
    merged, train_raw, features_raw, stores_raw = feat.load_and_merge_data('data')

    # 1. Save EDA tables
    # Missing values audit
    missing_summary = pd.DataFrame({
        'Dataset': ['train.csv', 'features.csv', 'stores.csv', 'merged_store_level'],
        'Total_Rows': [len(train_raw), len(features_raw), len(stores_raw), len(merged)],
        'Total_Columns': [train_raw.shape[1], features_raw.shape[1], stores_raw.shape[1], merged.shape[1]],
        'Date_Range': [
            f"{train_raw['Date'].min().strftime('%Y-%m-%d')} to {train_raw['Date'].max().strftime('%Y-%m-%d')}",
            f"{features_raw['Date'].min().strftime('%Y-%m-%d')} to {features_raw['Date'].max().strftime('%Y-%m-%d')}",
            "N/A (Store metadata)",
            f"{merged['Date'].min().strftime('%Y-%m-%d')} to {merged['Date'].max().strftime('%Y-%m-%d')}"
        ],
        'Null_Cells': [
            train_raw.isnull().sum().sum(),
            features_raw.isnull().sum().sum(),
            stores_raw.isnull().sum().sum(),
            merged.isnull().sum().sum()
        ]
    })
    missing_summary.to_csv(os.path.join(TABLES_DIR, 'missing_values_audit.csv'), index=False)
    print("Saved missing_values_audit.csv")

    # Clean markdown
    df_clean = feat.clean_markdown_and_outliers(merged)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])

    # Overall weekly total sales aggregation
    weekly_total = df_clean.groupby('Date').agg({
        'Weekly_Sales': 'sum',
        'IsHoliday': 'max',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'CPI': 'mean',
        'Unemployment': 'mean'
    }).reset_index()

    # Summary statistics table
    stats_df = df_clean[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']].describe().T
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
    stats_df.to_csv(os.path.join(TABLES_DIR, 'eda_summary_statistics.csv'))
    print("Saved eda_summary_statistics.csv")

    # -------------------------------------------------------------
    # PLOT 1: Overall Sales Trend
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(weekly_total['Date'], weekly_total['Weekly_Sales'] / 1e6, color='#1f77b4', lw=2, label='Total Weekly Sales (All 45 Stores)')
    
    # Highlight holiday weeks
    holiday_dates = weekly_total[weekly_total['IsHoliday'] == 1]
    ax.scatter(holiday_dates['Date'], holiday_dates['Weekly_Sales'] / 1e6, color='#d62728', s=60, zorder=5, label='Holiday Weeks')

    # Annotate key peaks (Thanksgiving / Black Friday)
    for _, row in holiday_dates.iterrows():
        # Thanksgiving is in late November
        if row['Date'].month == 11:
            ax.annotate('Thanksgiving / Black Friday Peak',
                        xy=(row['Date'], row['Weekly_Sales'] / 1e6),
                        xytext=(row['Date'] - pd.Timedelta(days=70), (row['Weekly_Sales'] / 1e6) + 3),
                        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                        fontsize=9, fontweight='bold', color='#333333')
        elif row['Date'].month == 12:
            ax.annotate('Pre-Christmas Peak',
                        xy=(row['Date'], row['Weekly_Sales'] / 1e6),
                        xytext=(row['Date'] + pd.Timedelta(days=20), (row['Weekly_Sales'] / 1e6) + 2),
                        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                        fontsize=9, fontweight='bold', color='#333333')

    ax.set_title('Walmart Chain-Wide Weekly Sales Trend (2010 - 2012)', fontsize=15, pad=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Weekly Sales ($ Millions)', fontsize=12)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'overall_sales_trend.png'), dpi=300)
    plt.close()
    print("Saved overall_sales_trend.png")

    # -------------------------------------------------------------
    # PLOT 2: Seasonality by Week of Year and Month
    # -------------------------------------------------------------
    df_clean['week_of_year'] = df_clean['Date'].dt.isocalendar().week.astype(int)
    df_clean['month'] = df_clean['Date'].dt.month

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Weekly seasonality
    week_agg = df_clean.groupby('week_of_year')['Weekly_Sales'].mean() / 1e3
    ax1.plot(week_agg.index, week_agg.values, marker='o', color='#2ca02c', lw=2, markersize=5)
    ax1.fill_between(week_agg.index, week_agg.values, color='#2ca02c', alpha=0.15)
    ax1.set_title('Average Weekly Sales by Week of Year', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Week of Year (1 - 52)', fontsize=11)
    ax1.set_ylabel('Mean Store Weekly Sales ($ Thousands)', fontsize=11)
    ax1.axvline(x=47, color='#d62728', linestyle='--', alpha=0.7, label='Week 47 (Thanksgiving)')
    ax1.axvline(x=51, color='#9467bd', linestyle='--', alpha=0.7, label='Week 51 (Christmas)')
    ax1.legend()

    # Monthly seasonality boxplot
    sns.boxplot(data=df_clean, x='month', y='Weekly_Sales', ax=ax2, palette='Blues', showfliers=False)
    ax2.set_title('Store Weekly Sales Distribution by Month', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Month (1 = Jan, 12 = Dec)', fontsize=11)
    ax2.set_ylabel('Weekly Sales ($)', fontsize=11)
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'seasonality_weekly_monthly.png'), dpi=300)
    plt.close()
    print("Saved seasonality_weekly_monthly.png")

    # -------------------------------------------------------------
    # PLOT 3: Holiday vs Non-Holiday Sales
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df_clean, x='IsHoliday', y='Weekly_Sales', ax=ax1, palette=['#4c72b0', '#c44e52'], showfliers=False)
    ax1.set_title('Weekly Sales: Holiday vs Non-Holiday Weeks', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(['Non-Holiday (0)', 'Holiday (1)'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Store Weekly Sales ($)', fontsize=11)

    mean_holiday = df_clean.groupby('IsHoliday')['Weekly_Sales'].mean()
    uplift_pct = ((mean_holiday[1] - mean_holiday[0]) / mean_holiday[0]) * 100
    bars = ax2.bar(['Non-Holiday', 'Holiday'], mean_holiday.values / 1e3, color=['#4c72b0', '#c44e52'], width=0.5)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 10, f"${b.get_height():,.1f}K", ha='center', va='bottom', fontweight='bold')
    ax2.set_title(f'Mean Sales Comparison (Uplift: +{uplift_pct:.2f}%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Store Weekly Sales ($ Thousands)', fontsize=11)
    ax2.set_ylim(0, max(mean_holiday.values / 1e3) * 1.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'holiday_vs_non_holiday_sales.png'), dpi=300)
    plt.close()
    print("Saved holiday_vs_non_holiday_sales.png")

    # -------------------------------------------------------------
    # PLOT 4: Sales by Store Type
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    type_palette = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    sns.boxplot(data=df_clean, x='Type', y='Weekly_Sales', order=['A', 'B', 'C'], palette=type_palette, ax=ax1, showfliers=False)
    ax1.set_title('Store Weekly Sales Distribution by Store Type', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Store Type (A = Supercenter, B = Medium, C = Small)', fontsize=11)
    ax1.set_ylabel('Weekly Sales ($)', fontsize=11)

    type_summary = df_clean.groupby('Type').agg({'Weekly_Sales': 'sum', 'Store': 'nunique'}).reset_index()
    type_summary['Pct_Total'] = (type_summary['Weekly_Sales'] / type_summary['Weekly_Sales'].sum()) * 100
    ax2.pie(type_summary['Weekly_Sales'], labels=[f"Type {t} ({p:.1f}%)\n{n} Stores" for t, p, n in zip(type_summary['Type'], type_summary['Pct_Total'], type_summary['Store'])],
            colors=[type_palette[t] for t in type_summary['Type']], autopct='%1.1f%%', startangle=140, textprops={'fontweight': 'bold'})
    ax2.set_title('Total Revenue Share by Store Type', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'sales_by_store_type.png'), dpi=300)
    plt.close()
    print("Saved sales_by_store_type.png")

    # -------------------------------------------------------------
    # PLOT 5: Correlation Heatmap
    # -------------------------------------------------------------
    corr_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday']
    corr_matrix = df_clean[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Sales and External Features', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("Saved correlation_heatmap.png")

    # -------------------------------------------------------------
    # PLOT 6: Sales vs External Factors
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    factors = [
        ('Temperature', 'Temperature (°F)', '#e74c3c'),
        ('Fuel_Price', 'Fuel Price ($/gal)', '#f39c12'),
        ('CPI', 'Consumer Price Index (CPI)', '#2980b9'),
        ('Unemployment', 'Unemployment Rate (%)', '#8e44ad')
    ]

    for ax, (col, label, color) in zip(axes.flatten(), factors):
        sns.regplot(data=df_clean.sample(1500, random_state=42), x=col, y='Weekly_Sales', ax=ax,
                    scatter_kws={'alpha': 0.25, 'color': color, 's': 15}, line_kws={'color': 'black', 'lw': 1.8})
        ax.set_title(f'Sales vs {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Weekly Sales ($)', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'sales_vs_external_factors.png'), dpi=300)
    plt.close()
    print("Saved sales_vs_external_factors.png")

    # -------------------------------------------------------------
    # STL DECOMPOSITION & ADF TEST
    # -------------------------------------------------------------
    # Set Date as index with weekly frequency
    ts_total = weekly_total.set_index('Date')['Weekly_Sales']
    ts_total = ts_total.asfreq('W-FRI')

    stl = STL(ts_total, period=52, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(ts_total.index, ts_total.values / 1e6, color='#1f77b4', lw=1.8)
    axes[0].set_ylabel('Observed ($M)', fontweight='bold')
    axes[0].set_title('STL Decomposition: Walmart Aggregate Weekly Sales (Period = 52 Weeks)', fontsize=14, fontweight='bold')

    axes[1].plot(res.trend.index, res.trend.values / 1e6, color='#ff7f0e', lw=2)
    axes[1].set_ylabel('Trend ($M)', fontweight='bold')

    axes[2].plot(res.seasonal.index, res.seasonal.values / 1e6, color='#2ca02c', lw=1.8)
    axes[2].set_ylabel('Seasonal ($M)', fontweight='bold')

    axes[3].scatter(res.resid.index, res.resid.values / 1e6, color='#d62728', s=18, alpha=0.7)
    axes[3].axhline(0, color='black', linestyle='--', lw=1)
    axes[3].set_ylabel('Residual ($M)', fontweight='bold')
    axes[3].set_xlabel('Date', fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'stl_decomposition.png'), dpi=300)
    plt.close()
    print("Saved stl_decomposition.png")

    # ADF Stationarity Test
    adf_raw = adfuller(ts_total.dropna(), autolag='AIC')
    diff_ts = ts_total.diff().dropna()
    adf_diff = adfuller(diff_ts, autolag='AIC')

    adf_results = pd.DataFrame({
        'Series': ['Raw Total Sales', 'First Difference (\u0394 Sales)'],
        'ADF_Statistic': [adf_raw[0], adf_diff[0]],
        'p_value': [adf_raw[1], adf_diff[1]],
        'Lags_Used': [adf_raw[2], adf_diff[2]],
        'N_Observations': [adf_raw[3], adf_diff[3]],
        'Critical_Value_1%': [adf_raw[4]['1%'], adf_diff[4]['1%']],
        'Critical_Value_5%': [adf_raw[4]['5%'], adf_diff[4]['5%']],
        'Critical_Value_10%': [adf_raw[4]['10%'], adf_diff[4]['10%']],
        'Stationary_at_5%': [adf_raw[1] < 0.05, adf_diff[1] < 0.05]
    })
    adf_results.to_csv(os.path.join(TABLES_DIR, 'adf_test_results.csv'), index=False)
    print("Saved adf_test_results.csv")
    print("\nADF Test Results:\n", adf_results[['Series', 'ADF_Statistic', 'p_value', 'Stationary_at_5%']])

    # -------------------------------------------------------------
    # PLOT 8: TimeSeriesSplit Cross-Validation Visualization
    # -------------------------------------------------------------
    df_feat = feat.prepare_feature_pipeline(merged)
    train_df, test_df, split_date = feat.time_based_train_test_split(df_feat, test_weeks=16)
    
    unique_train_dates = np.array(sorted(train_df['Date'].unique()))
    tscv = TimeSeriesSplit(n_splits=5)

    fig, ax = plt.subplots(figsize=(14, 6))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(unique_train_dates)):
        train_d = unique_train_dates[train_idx]
        val_d = unique_train_dates[val_idx]
        
        ax.barh(y=f"Fold {fold+1}", left=train_d[0], width=train_d[-1] - train_d[0], height=0.5, color='#1f77b4', label='Training Origin' if fold == 0 else "")
        ax.barh(y=f"Fold {fold+1}", left=val_d[0], width=val_d[-1] - val_d[0], height=0.5, color='#ff7f0e', label='Validation Origin' if fold == 0 else "")

    test_dates = sorted(test_df['Date'].unique())
    ax.barh(y="Test Set", left=test_dates[0], width=test_dates[-1] - test_dates[0], height=0.5, color='#2ca02c', label='Holdout Test (16 wks)')

    ax.set_title('Expanding-Window TimeSeriesSplit Cross-Validation Schema (5 Folds)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Timeline', fontsize=12)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'timeseries_cv_folds.png'), dpi=300)
    plt.close()
    print("Saved timeseries_cv_folds.png")

    # Features metadata table
    feature_cols = feat.get_feature_columns(df_feat)
    features_meta = pd.DataFrame({
        'Feature_Name': feature_cols,
        'Type': [
            'Holiday Indicator' if 'Holiday' in c or 'is_' in c else
            'Calendar / Cyclical' if any(k in c for k in ['year', 'month', 'week', 'quarter']) else
            'Historical Lag' if 'lag' in c else
            'Rolling Statistic' if 'roll' in c else
            'Store Attribute' if any(k in c for k in ['Size', 'Type']) else
            'External Macro' for c in feature_cols
        ],
        'Leakage_Prevention': [
            'Shifted t-1' if any(k in c for k in ['lag', 'roll']) else
            'Pre-known Calendar / Exogenous' for c in feature_cols
        ]
    })
    features_meta.to_csv(os.path.join(TABLES_DIR, 'features_metadata.csv'), index=False)
    print("Saved features_metadata.csv")
    print("\nAll Part 1 figures and tables generated successfully!")

if __name__ == '__main__':
    run_eda_and_generate_artifacts()
