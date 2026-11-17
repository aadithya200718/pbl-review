"""
Evaluation Metrics and Statistical Hypothesis Testing for Walmart Sales Forecasting.

Includes:
- MAE, RMSE, MAPE, sMAPE, R2, MASE (scaled by seasonal naive benchmark)
- WMAE (Weighted MAE with holiday weeks weighted 5x per Kaggle competition rules)
- Separate holiday vs non-holiday MAE/RMSE
- Diebold-Mariano test with Harvey-Leybourne-Newbold small sample adjustment
"""

import numpy as np
import pandas as pd
from scipy import stats


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def r2_score_metric(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-8))


def weighted_mean_absolute_error(y_true, y_pred, is_holiday):
    """
    Walmart Kaggle competition official evaluation metric:
        WMAE = sum(w_i * |y_i - y_hat_i|) / sum(w_i)
    where w_i = 5 if is_holiday == 1 else 1.
    """
    weights = np.where(np.asarray(is_holiday) == 1, 5.0, 1.0)
    abs_errors = np.abs(y_true - y_pred)
    return np.sum(weights * abs_errors) / np.sum(weights)


def mean_absolute_scaled_error(mae, seasonal_naive_mae):
    """
    MASE = MAE / MAE_seasonal_naive
    MASE < 1 indicates the model outperforms the annual seasonal benchmark.
    """
    return mae / (seasonal_naive_mae + 1e-8)


def evaluate_forecast(y_true, y_pred, is_holiday=None, seasonal_naive_mae=None, fit_time_sec=0.0, n_bases=0):
    """
    Computes all standard and competition evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score_metric(y_true, y_pred)

    # WMAE and holiday breakdown
    if is_holiday is not None:
        is_hol = np.asarray(is_holiday)
        wmae = weighted_mean_absolute_error(y_true, y_pred, is_hol)

        hol_mask = (is_hol == 1)
        non_hol_mask = (is_hol == 0)

        mae_holiday = mean_absolute_error(y_true[hol_mask], y_pred[hol_mask]) if hol_mask.any() else np.nan
        mae_non_holiday = mean_absolute_error(y_true[non_hol_mask], y_pred[non_hol_mask]) if non_hol_mask.any() else np.nan
        rmse_holiday = root_mean_squared_error(y_true[hol_mask], y_pred[hol_mask]) if hol_mask.any() else np.nan
        rmse_non_holiday = root_mean_squared_error(y_true[non_hol_mask], y_pred[non_hol_mask]) if non_hol_mask.any() else np.nan
    else:
        wmae = mae
        mae_holiday = np.nan
        mae_non_holiday = np.nan
        rmse_holiday = np.nan
        rmse_non_holiday = np.nan

    mase = mean_absolute_scaled_error(mae, seasonal_naive_mae) if seasonal_naive_mae is not None else np.nan

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'sMAPE (%)': smape,
        'R2': r2,
        'MASE': mase,
        'WMAE': wmae,
        'Holiday_MAE': mae_holiday,
        'NonHoliday_MAE': mae_non_holiday,
        'Holiday_RMSE': rmse_holiday,
        'NonHoliday_RMSE': rmse_non_holiday,
        'Train_Time_Sec': fit_time_sec,
        'N_Bases_Parameters': n_bases
    }
    return metrics


def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion='squared'):
    """
    Computes the Diebold-Mariano test statistic for comparing forecast accuracy of two models.
    H0: Both models have equal forecast accuracy.
    H1: Model 2 is significantly more accurate than Model 1.
    Applies Harvey-Leybourne-Newbold (HLN, 1997) correction for small-sample finite horizon bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    e1 = y_true - np.asarray(y_pred1, dtype=float)
    e2 = y_true - np.asarray(y_pred2, dtype=float)

    if criterion == 'squared':
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)

    n = len(d)
    mean_d = np.mean(d)

    # Autocovariance up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        gamma_sum += (1.0 - k / h) * gamma_k

    var_d = (gamma_0 + 2.0 * gamma_sum) / n
    if var_d <= 0:
        dm_stat = 0.0
        p_val = 1.0
    else:
        dm_stat = mean_d / np.sqrt(var_d)
        # HLN adjustment factor
        hln_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
        dm_stat = dm_stat * hln_factor
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        'DM_Statistic': dm_stat,
        'p_value': p_val,
        'Mean_Loss_Diff': mean_d,
        'Significantly_Different_at_5%': p_val < 0.05
    }
