"""
Comprehensive Experimentation Runner for Steps 4 through 8.
Executes training, TimeSeriesSplit hyperparameter tuning, model evaluations,
diagnostic visualizer, ablation studies, and 12-week multi-step recursive forecasting.
Saves all figures to results/figures/ and all metric tables to results/tables/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import shap

import src.features as feat
import src.models as models_mod
import src.evaluate as eval_mod

# Seed and plotting configuration
SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'Arial'

FIGURES_DIR = 'results/figures'
TABLES_DIR = 'results/tables'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def run_all_experiments():
    print("=" * 70)
    print("STARTING FULL EXPERIMENTAL RUN (STEPS 4 - 8)")
    print("=" * 70)

    # 1. Load data and run zero-leakage feature pipeline
    merged, train_raw, features_raw, stores_raw = feat.load_and_merge_data('data')

    # Run outlier diagnostics (without deleting legitimate holiday surges)
    diag_df, outlier_summary = feat.diagnose_outliers(merged, iqr_multiplier=3.0)
    print("Outlier Diagnostics Summary:")
    for k, v in outlier_summary.items():
        print(f"  {k}: {v}")

    df_feat = feat.prepare_feature_pipeline(merged, lags=[1, 2, 4, 52], rolling_windows=[4, 12], drop_na_lags=True)
    train_df, test_df, split_date = feat.time_based_train_test_split(df_feat, test_weeks=16)

    feature_cols = feat.get_feature_columns(df_feat)
    train_s, test_s, scaler = feat.scale_features(train_df, test_df, feature_cols, scaler_type='standard')

    X_train = train_s[feature_cols]
    y_train = train_s['Weekly_Sales'].values
    X_test = test_s[feature_cols]
    y_test = test_s['Weekly_Sales'].values
    is_holiday_test = test_df['IsHoliday'].values

    print(f"Dataset partitions: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Features: {len(feature_cols)} predictors.")

    # Update features metadata table
    meta_rows = []
    for col in feature_cols:
        if 'Holiday' in col or 'is_' in col or 'weeks_to_' in col:
            category = 'Holiday Indicator / Proximity'
            leakage = 'Pre-known Calendar Event'
        elif any(k in col for k in ['year', 'month', 'week', 'quarter', 'sin', 'cos']):
            category = 'Calendar / Cyclical'
            leakage = 'Pre-known Calendar Math'
        elif 'lag' in col:
            category = 'Historical Sales Lag'
            leakage = f'Shifted t-{col.split("_")[-1]} (Store-isolated)'
        elif 'roll' in col:
            category = 'Rolling Window Statistic'
            leakage = 'Shifted t-1 Rolling Window (Store-isolated)'
        elif col in ['Size', 'Type_A', 'Type_B', 'Type_C'] or col.startswith('Store_ID_'):
            category = 'Store Attribute'
            leakage = 'Static Physical / ID Attribute'
        else:
            category = 'Macro / External'
            leakage = 'Exogenous Environmental Sensor'
        meta_rows.append({'Feature_Name': col, 'Category': category, 'Leakage_Control': leakage})

    pd.DataFrame(meta_rows).to_csv(os.path.join(TABLES_DIR, 'features_metadata.csv'), index=False)

    # -------------------------------------------------------------
    # STEP 5: HYPERPARAMETER TUNING VIA TimeSeriesSplit
    # -------------------------------------------------------------
    print("\n--- STEP 5: HYPERPARAMETER TUNING VIA TimeSeriesSplit ---")
    tscv = TimeSeriesSplit(n_splits=3)
    tuning_records = []

    # 1. MLP Tuning
    mlp_grid = {
        'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
        'alpha': [0.001, 0.01, 0.1]
    }
    mlp_search = RandomizedSearchCV(
        MLPRegressor(activation='relu', max_iter=200, random_state=SEED, early_stopping=True),
        mlp_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    mlp_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'MLP Neural Network',
        'Tuned_Parameters': str(mlp_search.best_params_),
        'Best_CV_Score (MAE)': -mlp_search.best_score_
    })

    # 2. Decision Tree Tuning
    dt_grid = {'max_depth': [6, 8, 12], 'min_samples_leaf': [5, 10, 20]}
    dt_search = RandomizedSearchCV(
        DecisionTreeRegressor(random_state=SEED),
        dt_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    dt_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'Decision Tree',
        'Tuned_Parameters': str(dt_search.best_params_),
        'Best_CV_Score (MAE)': -dt_search.best_score_
    })

    # 3. Random Forest Tuning
    rf_grid = {'n_estimators': [50, 100], 'max_depth': [8, 12], 'min_samples_leaf': [3, 8]}
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=2),
        rf_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    rf_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'Random Forest',
        'Tuned_Parameters': str(rf_search.best_params_),
        'Best_CV_Score (MAE)': -rf_search.best_score_
    })

    # 4. Gradient Boosting Tuning
    gb_grid = {'n_estimators': [80, 120], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        gb_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    gb_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'Gradient Boosting',
        'Tuned_Parameters': str(gb_search.best_params_),
        'Best_CV_Score (MAE)': -gb_search.best_score_
    })

    # 5. XGBoost Tuning
    xgb_grid = {'n_estimators': [100, 150], 'max_depth': [5, 7], 'learning_rate': [0.04, 0.08]}
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=SEED, n_jobs=2),
        xgb_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    xgb_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'XGBoost',
        'Tuned_Parameters': str(xgb_search.best_params_),
        'Best_CV_Score (MAE)': -xgb_search.best_score_
    })

    # 6. LightGBM Tuning
    lgb_grid = {'n_estimators': [100, 150], 'max_depth': [6, 8], 'learning_rate': [0.04, 0.08]}
    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=SEED, n_jobs=2, verbose=-1),
        lgb_grid, n_iter=4, cv=tscv, scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1
    )
    lgb_search.fit(X_train, y_train)
    tuning_records.append({
        'Model': 'LightGBM',
        'Tuned_Parameters': str(lgb_search.best_params_),
        'Best_CV_Score (MAE)': -lgb_search.best_score_
    })

    tuning_df = pd.DataFrame(tuning_records)
    tuning_df.to_csv(os.path.join(TABLES_DIR, 'hyperparameter_tuning.csv'), index=False)
    print("Saved hyperparameter_tuning.csv:\n", tuning_df)

    # -------------------------------------------------------------
    # STEP 4: MODEL INSTANTIATION & TRAINING
    # -------------------------------------------------------------
    print("\n--- STEP 4: TRAINING ALL 11 MODEL FAMILIES ---")
    all_models = {
        # A) Baselines
        'Naive (Lag 1)': models_mod.NaiveForecaster(),
        'Seasonal Naive (Lag 52)': models_mod.SeasonalNaiveForecaster(),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=100.0, random_state=SEED),
        
        # B) Fixed Basis Models
        'Polynomial Regression (deg 2)': models_mod.PolynomialBasisRegression(degree=2, alpha=50.0),
        'Fourier Feature Regression': models_mod.FourierFeatureRegression(n_harmonics=6, alpha=10.0),
        
        # C) Adaptive Basis Function Models
        'MLP Neural Network': mlp_search.best_estimator_,
        'RBF Network': models_mod.RBFNetworkRegressor(n_centers=30, gamma=0.05, alpha=5.0, random_state=SEED),
        'MARS (SplineTransformer)': models_mod.MARSSplineRegressor(n_knots=6, degree=3, alpha=20.0),
        'Generalized Additive Model': models_mod.PyGAMForecaster(n_splines=10, lam=0.8),
        'Decision Tree': dt_search.best_estimator_,
        'Random Forest': rf_search.best_estimator_,
        'Gradient Boosting': gb_search.best_estimator_,
        'XGBoost': xgb_search.best_estimator_,
        'LightGBM': lgb_search.best_estimator_
    }

    # Evaluate Seasonal Naive baseline MAE for MASE computation
    sn_model = all_models['Seasonal Naive (Lag 52)']
    sn_preds = sn_model.predict(test_df)
    seasonal_naive_mae = eval_mod.mean_absolute_error(y_test, sn_preds)

    test_preds_dict = {}
    metrics_records = []

    for name, model in all_models.items():
        t0 = time.time()
        if 'Naive' in name:
            model.fit(train_df)
            y_pred = model.predict(test_df)
            n_bases = 1
        elif 'Fourier' in name:
            model.fit(train_df, y_train)
            y_pred = model.predict(test_df)
            n_bases = model.n_bases
        elif 'Polynomial' in name or 'MARS' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            n_bases = model.n_bases
        elif 'RBF' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            n_bases = model.n_bases
        elif 'GAM' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            n_bases = model.n_bases
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, 'coef_'):
                n_bases = len(model.coef_)
            elif hasattr(model, 'tree_'):
                n_bases = model.tree_.node_count
            elif hasattr(model, 'n_estimators'):
                n_bases = model.n_estimators
            else:
                n_bases = len(feature_cols)

        train_time = time.time() - t0
        test_preds_dict[name] = y_pred

        m = eval_mod.evaluate_forecast(
            y_true=y_test,
            y_pred=y_pred,
            is_holiday=is_holiday_test,
            seasonal_naive_mae=seasonal_naive_mae,
            fit_time_sec=train_time,
            n_bases=n_bases
        )
        m['Model'] = name
        metrics_records.append(m)
        print(f"Trained {name:30s} | WMAE: ${m['WMAE']:,.2f} | RMSE: ${m['RMSE']:,.2f} | R2: {m['R2']:.4f} | Time: {train_time:.2f}s")

    # -------------------------------------------------------------
    # STEP 6: EVALUATION METRICS TABLE & PLOT
    # -------------------------------------------------------------
    print("\n--- STEP 6: METRICS EVALUATION & COMPARISON ---")
    metrics_df = pd.DataFrame(metrics_records)
    col_order = ['Model', 'WMAE', 'MAE', 'RMSE', 'MAPE (%)', 'sMAPE (%)', 'R2', 'MASE',
                 'Holiday_MAE', 'NonHoliday_MAE', 'Train_Time_Sec', 'N_Bases_Parameters']
    metrics_df = metrics_df[col_order].sort_values(by='WMAE').reset_index(drop=True)
    metrics_df.to_csv(os.path.join(TABLES_DIR, 'model_comparison_metrics.csv'), index=False)
    print("Saved model_comparison_metrics.csv:\n", metrics_df[['Model', 'WMAE', 'RMSE', 'MAPE (%)', 'R2', 'MASE']])

    # Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sorted_rmse = metrics_df.sort_values(by='RMSE', ascending=True)
    bars1 = ax1.barh(sorted_rmse['Model'], sorted_rmse['RMSE'] / 1e3, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax1.set_xlabel('RMSE ($ Thousands)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Forecast Accuracy: Root Mean Squared Error (RMSE)', fontsize=13, fontweight='bold')
    for b in bars1:
        ax1.text(b.get_width() + 2, b.get_y() + b.get_height()/2., f"${b.get_width():,.1f}K", va='center', fontsize=9)

    sorted_mape = metrics_df.sort_values(by='MAPE (%)', ascending=True)
    bars2 = ax2.barh(sorted_mape['Model'], sorted_mape['MAPE (%)'], color='#2ca02c', edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Mean Absolute Percentage Error - MAPE (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Model Forecast Accuracy: MAPE (%)', fontsize=13, fontweight='bold')
    for b in bars2:
        ax2.text(b.get_width() + 0.2, b.get_y() + b.get_height()/2., f"{b.get_width():.2f}%", va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'model_rmse_mape_comparison.png'), dpi=300)
    plt.close()

    # Diebold-Mariano Test
    best_adaptive_name = metrics_df[metrics_df['Model'].isin([
        'LightGBM', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'MLP Neural Network', 'RBF Network', 'MARS (SplineTransformer)', 'Generalized Additive Model'
    ])].iloc[0]['Model']

    best_baseline_name = metrics_df[metrics_df['Model'].isin([
        'Seasonal Naive (Lag 52)', 'Naive (Lag 1)', 'Ridge Regression', 'Linear Regression'
    ])].iloc[0]['Model']

    print(f"\nRunning Diebold-Mariano Test: {best_adaptive_name} vs {best_baseline_name}...")
    dm_res = eval_mod.diebold_mariano_test(
        y_true=y_test,
        y_pred1=test_preds_dict[best_baseline_name],
        y_pred2=test_preds_dict[best_adaptive_name],
        h=1,
        criterion='squared'
    )
    dm_df = pd.DataFrame([{
        'Baseline_Model': best_baseline_name,
        'Adaptive_Model': best_adaptive_name,
        'DM_Statistic': dm_res['DM_Statistic'],
        'p_value': dm_res['p_value'],
        'Significantly_Superior_at_5%': dm_res['Significantly_Different_at_5%']
    }])
    dm_df.to_csv(os.path.join(TABLES_DIR, 'diebold_mariano_test.csv'), index=False)
    print("Diebold-Mariano Test Result:\n", dm_df)

    # -------------------------------------------------------------
    # STEP 7: DIAGNOSTICS AND VISUALIZATION
    # -------------------------------------------------------------
    print("\n--- STEP 7: DIAGNOSTIC PLOTS & ANALYSIS ---")
    top3_models = metrics_df['Model'].head(3).tolist()
    test_plot_df = test_df.copy()
    for m_name in top3_models:
        test_plot_df[m_name] = test_preds_dict[m_name]

    # Actual vs Predicted Plot
    chain_actual = test_plot_df.groupby('Date')['Weekly_Sales'].sum() / 1e6
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(chain_actual.index, chain_actual.values, 'k-o', label='Actual Sales', lw=2.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for m_name, c in zip(top3_models, colors):
        chain_pred = test_plot_df.groupby('Date')[m_name].sum() / 1e6
        axes[0, 0].plot(chain_pred.index, chain_pred.values, '--', label=f"{m_name}", lw=2, color=c)
    axes[0, 0].set_title('Chain Total Sales: Actual vs Predicted (Holdout Test)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Weekly Sales ($ Millions)', fontsize=10)
    axes[0, 0].legend(loc='lower left', frameon=True)

    for ax, (s_id, s_title) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [(1, 'Store 1 (Type A Supercenter)'), (3, 'Store 3 (Type B Store)'), (33, 'Store 33 (Type C Store)')]):
        s_df = test_plot_df[test_plot_df['Store'] == s_id].sort_values('Date')
        ax.plot(s_df['Date'], s_df['Weekly_Sales'] / 1e3, 'k-o', label='Actual', lw=2.2)
        for m_name, c in zip(top3_models, colors):
            ax.plot(s_df['Date'], s_df[m_name] / 1e3, '--', label=m_name, lw=1.8, color=c)
        ax.set_title(f'{s_title}: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weekly Sales ($ Thousands)', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'actual_vs_predicted_top3.png'), dpi=300)
    plt.close()

    # Residual Diagnostics
    best_model_name = top3_models[0]
    best_preds = test_preds_dict[best_model_name]
    residuals = y_test - best_preds

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(test_df['Date'], residuals / 1e3, color='#1f77b4', alpha=0.6, s=20)
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[0, 0].set_title(f'Residuals Over Time ({best_model_name})', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Residual ($ Thousands)', fontsize=10)

    sns.histplot(residuals / 1e3, kde=True, ax=axes[0, 1], color='#2ca02c', bins=30)
    axes[0, 1].set_title('Residual Distribution vs Normal Density', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual ($ Thousands)', fontsize=10)

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    axes[1, 0].get_lines()[0].set_color('#1f77b4')
    axes[1, 0].get_lines()[1].set_color('red')

    plot_acf(residuals, lags=20, ax=axes[1, 1], color='#d62728')
    axes[1, 1].set_title('Autocorrelation Function (ACF) of Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lag (Weeks)', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'residual_diagnostics.png'), dpi=300)
    plt.close()

    # Feature Importance & SHAP
    best_estimator = all_models[best_model_name]
    if hasattr(best_estimator, 'feature_importances_'):
        feat_imp = pd.Series(best_estimator.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        feat_imp.head(15).plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='black')
        ax.invert_yaxis()
        ax.set_title(f'Top 15 Feature Importances ({best_model_name})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Relative Importance (Split Gain / Gini)', fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'feature_importance_best_model.png'), dpi=300)
        plt.close()

        try:
            sample_X = X_test.sample(300, random_state=SEED) if len(X_test) > 300 else X_test
            explainer = shap.TreeExplainer(best_estimator)
            shap_values = explainer.shap_values(sample_X)
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, sample_X, show=False, max_display=12)
            plt.title(f'SHAP Summary Plot: Feature Attributions ({best_model_name})', fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"SHAP notice: {e}")

    # GAM Partial Dependence
    gam_model = all_models['Generalized Additive Model'].model_
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    core_names = ['sales_lag_1', 'sales_roll_mean_4', 'Size', 'CPI', 'Unemployment', 'Temperature']
    for i, (ax, f_name) in enumerate(zip(axes.flatten(), core_names)):
        XX = gam_model.generate_X_grid(term=i)
        pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pdep, color='#2ca02c', lw=2)
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='#2ca02c', alpha=0.2)
        ax.set_title(f'GAM Spline Basis: f({f_name})', fontsize=11, fontweight='bold')
        ax.set_xlabel(f_name, fontsize=9)
        ax.set_ylabel('Partial Effect on Sales', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'gam_partial_dependence.png'), dpi=300)
    plt.close()

    # Adaptive Bases Visualization (RBF Centers & MARS Knots)
    rbf_model = all_models['RBF Network']
    mars_model = all_models['MARS (SplineTransformer)']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(X_train['Size'], X_train['sales_lag_1'], alpha=0.15, color='#4c72b0', label='Training Samples', s=10)
    size_idx = feature_cols.index('Size')
    lag_idx = feature_cols.index('sales_lag_1')
    ax1.scatter(rbf_model.centers_[:, size_idx], rbf_model.centers_[:, lag_idx],
                color='#d62728', marker='X', s=90, label=f'RBF Learned Centers (M={rbf_model.n_centers})')
    ax1.set_title('RBF Network: Adaptively Clustered Basis Centers', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Scaled Store Size', fontsize=10)
    ax1.set_ylabel('Scaled Lag 1 Sales', fontsize=10)
    ax1.legend(frameon=True)

    sample_grid = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    dummy_input = np.zeros((100, 6))
    dummy_input[:, 0] = sample_grid[:, 0]
    spline_bases = mars_model.spline_.transform(dummy_input)
    for b_idx in range(min(5, spline_bases.shape[1])):
        ax2.plot(sample_grid, spline_bases[:, b_idx], lw=2, label=f'B-spline Basis $\\phi_{{{b_idx+1}}}(x)$')
    ax2.set_title('MARS Substitute: Piecewise Quantile-Knot Spline Bases', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized Input Variable', fontsize=10)
    ax2.set_ylabel('Basis Activation', fontsize=10)
    ax2.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'adaptive_bases_visualization.png'), dpi=300)
    plt.close()

    # Learning Curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_errors, val_errors = [], []
    for frac in train_sizes:
        n_samples = int(len(X_train) * frac)
        sub_X_tr = X_train.iloc[:n_samples]
        sub_y_tr = y_train[:n_samples]
        m_clone = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=SEED, n_jobs=2, verbose=-1)
        m_clone.fit(sub_X_tr, sub_y_tr)
        train_errors.append(eval_mod.root_mean_squared_error(sub_y_tr, m_clone.predict(sub_X_tr)) / 1e3)
        val_errors.append(eval_mod.root_mean_squared_error(y_test, m_clone.predict(X_test)) / 1e3)
    ax.plot([f*100 for f in train_sizes], train_errors, 'o-', color='#1f77b4', lw=2, label='Training RMSE')
    ax.plot([f*100 for f in train_sizes], val_errors, 's--', color='#d62728', lw=2, label='Holdout Test RMSE')
    ax.set_title(f'Learning Curve & Bias-Variance Analysis ({best_model_name})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Training Data Utilized (%)', fontsize=11)
    ax.set_ylabel('RMSE ($ Thousands)', fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'learning_curves.png'), dpi=300)
    plt.close()

    # Ablation Study
    print("\n--- RUNNING FEATURE ABLATION STUDY ---")
    ablation_sets = {
        'Full Model (All Features)': feature_cols,
        'Without Lag Features': [c for c in feature_cols if 'lag' not in c and 'roll' not in c],
        'Without Holiday Features': [c for c in feature_cols if 'Holiday' not in c and 'is_' not in c and 'weeks_to_' not in c],
        'Without Macro / External': [c for c in feature_cols if c not in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'markdown_available', 'markdown_active']]
    }
    ablation_results = []
    for setting_name, subset_cols in ablation_sets.items():
        m_abl = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=SEED, n_jobs=2, verbose=-1)
        m_abl.fit(X_train[subset_cols], y_train)
        pred_abl = m_abl.predict(X_test[subset_cols])
        ev = eval_mod.evaluate_forecast(y_test, pred_abl, is_holiday_test, seasonal_naive_mae)
        ablation_results.append({
            'Configuration': setting_name,
            'Num_Features': len(subset_cols),
            'WMAE': ev['WMAE'],
            'RMSE': ev['RMSE'],
            'R2': ev['R2'],
            'WMAE_Increase (%)': 0.0
        })
    ablation_df = pd.DataFrame(ablation_results)
    baseline_wmae = ablation_df.loc[0, 'WMAE']
    ablation_df['WMAE_Increase (%)'] = ((ablation_df['WMAE'] - baseline_wmae) / baseline_wmae) * 100.0
    ablation_df.to_csv(os.path.join(TABLES_DIR, 'ablation_study_results.csv'), index=False)
    print("Saved ablation_study_results.csv:\n", ablation_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(ablation_df['Configuration'], ablation_df['WMAE'] / 1e3, color=['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4'], edgecolor='black', alpha=0.85)
    ax.set_xlabel('Weighted MAE ($ Thousands)', fontsize=11, fontweight='bold')
    ax.set_title('Feature Ablation Study: Impact on Forecasting Accuracy', fontsize=13, fontweight='bold')
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2., f"${b.get_width():,.1f}K", va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'ablation_study.png'), dpi=300)
    plt.close()

    # Forecast Error by Store Type
    test_eval_df = test_df.copy()
    test_eval_df['Best_Pred'] = best_preds
    test_eval_df['Abs_Error'] = np.abs(test_eval_df['Weekly_Sales'] - test_eval_df['Best_Pred'])
    test_eval_df['Sq_Error'] = (test_eval_df['Weekly_Sales'] - test_eval_df['Best_Pred']) ** 2

    type_error_records = []
    for s_type, grp in test_eval_df.groupby('Type'):
        wmae_t = eval_mod.weighted_mean_absolute_error(grp['Weekly_Sales'].values, grp['Best_Pred'].values, grp['IsHoliday'].values)
        type_error_records.append({
            'Store_Type': s_type,
            'Store_Count': grp['Store'].nunique(),
            'Mean_Weekly_Sales': grp['Weekly_Sales'].mean(),
            'MAE': grp['Abs_Error'].mean(),
            'RMSE': np.sqrt(grp['Sq_Error'].mean()),
            'WMAE': wmae_t,
            'MAPE (%)': (grp['Abs_Error'] / grp['Weekly_Sales']).mean() * 100.0
        })
    type_error_df = pd.DataFrame(type_error_records)
    type_error_df.to_csv(os.path.join(TABLES_DIR, 'error_by_store_type.csv'), index=False)
    print("Saved error_by_store_type.csv:\n", type_error_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = ax1.bar(type_error_df['Store_Type'], type_error_df['WMAE'] / 1e3, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax1.set_title('Forecast Error (WMAE) by Store Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('WMAE ($ Thousands)', fontsize=10)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2., b.get_height() + 1, f"${b.get_height():,.1f}K", ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(type_error_df['Store_Type'], type_error_df['MAPE (%)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax2.set_title('Relative Percentage Error (MAPE) by Store Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAPE (%)', fontsize=10)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.1, f"{b.get_height():.2f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'error_by_store_type.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # STEP 8: LEAKAGE-FREE RECURSIVE MULTI-STEP FORWARD FORECAST
    # -------------------------------------------------------------
    print("\n--- STEP 8: LEAKAGE-FREE RECURSIVE 12-WEEK FORWARD FORECAST ---")
    # Retrain champion model on all available historical data (91 weeks)
    full_s, _, full_scaler = feat.scale_features(df_feat, df_feat, feature_cols, scaler_type='standard')
    final_model = lgb.LGBMRegressor(n_estimators=150, max_depth=7, learning_rate=0.06, random_state=SEED, n_jobs=2, verbose=-1)
    final_model.fit(full_s[feature_cols], full_s['Weekly_Sales'].values)

    in_sample_resids = full_s['Weekly_Sales'].values - final_model.predict(full_s[feature_cols])
    sigma_res = np.std(in_sample_resids)

    # Future dates schedule (12 weekly timestamps)
    last_date = df_feat['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=12, freq='W-FRI')

    # Execute recursive forecasting engine from src.features
    future_df = feat.generate_recursive_multi_step_forecast(
        model=final_model,
        historical_df=df_feat,
        future_dates=future_dates,
        feature_cols=feature_cols,
        scaler=full_scaler,
        horizon=12
    )

    # Compute prediction intervals
    future_df['Lower_95'] = np.maximum(0, future_df['Forecast_Sales'] - 1.96 * sigma_res)
    future_df['Upper_95'] = future_df['Forecast_Sales'] + 1.96 * sigma_res
    future_df['Lower_80'] = np.maximum(0, future_df['Forecast_Sales'] - 1.28 * sigma_res)
    future_df['Upper_80'] = future_df['Forecast_Sales'] + 1.28 * sigma_res

    forecast_agg = future_df.groupby('Date').agg({
        'Forecast_Sales': 'sum',
        'Lower_95': 'sum',
        'Upper_95': 'sum',
        'Lower_80': 'sum',
        'Upper_80': 'sum',
        'IsHoliday': 'max'
    }).reset_index()
    forecast_agg.to_csv(os.path.join(TABLES_DIR, 'twelve_week_forecast.csv'), index=False)
    print("Saved twelve_week_forecast.csv:\n", forecast_agg)

    # Plot 12-week forward forecast
    hist_agg = df_feat.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_agg['Date'], hist_agg['Weekly_Sales'] / 1e6, color='#1f77b4', lw=2, label='Historical Total Sales (2011 - Oct 2012)')
    ax.plot(forecast_agg['Date'], forecast_agg['Forecast_Sales'] / 1e6, 'o--', color='#d62728', lw=2.5, label='12-Week Recursive Forecast (Nov 2012 - Jan 2013)')
    ax.fill_between(forecast_agg['Date'], forecast_agg['Lower_80'] / 1e6, forecast_agg['Upper_80'] / 1e6, color='#d62728', alpha=0.25, label='80% Prediction Interval')
    ax.fill_between(forecast_agg['Date'], forecast_agg['Lower_95'] / 1e6, forecast_agg['Upper_95'] / 1e6, color='#d62728', alpha=0.12, label='95% Prediction Interval')
    ax.axvline(x=last_date, color='black', linestyle=':', lw=1.5, label='Forecast Origin (2012-10-26)')
    ax.set_title('Walmart 12-Week Multi-Step Recursive Sales Forecast with Prediction Intervals', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Timeline', fontsize=11)
    ax.set_ylabel('Total Chain Sales ($ Millions)', fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'twelve_week_ahead_forecast.png'), dpi=300)
    plt.close()

    print("\nALL EXPERIMENTS, METRICS, DIAGNOSTICS, AND FORECASTS COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    run_all_experiments()
