"""
Model Architectures for Sales Forecasting using Adaptive and Fixed Basis Function Models.

Every model conforms to the basis function representation:
    y_hat(x) = w_0 + sum_{j=1}^M w_j * phi_j(x; theta_j)

In fixed basis models (Polynomial, Fourier), phi_j is fixed independent of the training targets.
In adaptive basis models (MLP, RBF, MARS/Splines, GAM, Trees, Boosting), phi_j(x; theta_j)
is adaptively parameterized and optimized directly from data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from pygam import LinearGAM, s, f
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False


# ==============================================================================
# BASELINE 1 & 2: NAIVE AND SEASONAL NAIVE FORECASTERS
# ==============================================================================
class NaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Predicts the most recently observed value per store (Lag 1).
    Basis formulation: phi_1(x) = x_{lag_1}, w_1 = 1, w_0 = 0.
    """
    def __init__(self, lag_col='sales_lag_1'):
        self.lag_col = lag_col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.lag_col].values
        return X[:, 0]


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Predicts the value from 52 weeks ago (Lag 52).
    Captures annual retail seasonality with zero learning.
    Basis formulation: phi_1(x) = x_{lag_52}, w_1 = 1, w_0 = 0.
    """
    def __init__(self, lag_52_col='sales_lag_52'):
        self.lag_52_col = lag_52_col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.lag_52_col].values
        return X[:, 0]


# ==============================================================================
# BASELINE 3: SARIMA FORECASTER (ON AGGREGATE STORE SERIES)
# ==============================================================================
class AggregateSARIMAForecaster(BaseEstimator, RegressorMixin):
    """
    Fits SARIMAX(1, 1, 1)x(1, 0, 1, 52) on the aggregated chain-wide series,
    then apportions predictions back to individual stores based on historical store shares.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.store_weights_ = {}
        self.fitted_model_ = None
        self.train_dates_ = None

    def fit(self, train_df, y=None):
        store_totals = train_df.groupby('Store')['Weekly_Sales'].sum()
        self.store_weights_ = (store_totals / store_totals.sum()).to_dict()

        agg_chain = train_df.groupby('Date')['Weekly_Sales'].sum().sort_index()
        self.train_dates_ = agg_chain.index

        try:
            model = SARIMAX(
                agg_chain,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model_ = model.fit(disp=False, maxiter=40)
        except Exception:
            model = SARIMAX(agg_chain, order=(1, 1, 1), enforce_stationarity=False)
            self.fitted_model_ = model.fit(disp=False)
        return self

    def predict(self, test_df):
        n_steps = test_df['Date'].nunique()
        chain_preds = self.fitted_model_.forecast(steps=n_steps)
        date_pred_map = dict(zip(sorted(test_df['Date'].unique()), chain_preds))

        preds = []
        for _, row in test_df.iterrows():
            d = row['Date']
            s_id = row['Store']
            c_pred = date_pred_map.get(d, np.mean(chain_preds))
            w = self.store_weights_.get(s_id, 1.0 / 45.0)
            preds.append(c_pred * w)
        return np.array(preds)


# ==============================================================================
# FIXED BASIS MODEL 1: POLYNOMIAL REGRESSION
# ==============================================================================
class PolynomialBasisRegression(BaseEstimator, RegressorMixin):
    """
    Transforms continuous features using fixed polynomial expansions:
        phi_j(x) = x_i^d  (degree d in 2..4)
    Bases are fixed and data-independent; weights w_j are fitted via Ridge.
    """
    def __init__(self, degree=2, alpha=10.0):
        self.degree = degree
        self.alpha = alpha
        self.poly_ = PolynomialFeatures(degree=degree, include_bias=False)
        self.reg_ = Ridge(alpha=alpha, random_state=42)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            core_cols = [c for c in ['sales_lag_1', 'sales_roll_mean_4', 'Size', 'CPI', 'Unemployment', 'Temperature'] if c in X.columns]
            X_core = X[core_cols].values
        else:
            X_core = X[:, :6]
        X_poly = self.poly_.fit_transform(X_core)
        self.reg_.fit(X_poly, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            core_cols = [c for c in ['sales_lag_1', 'sales_roll_mean_4', 'Size', 'CPI', 'Unemployment', 'Temperature'] if c in X.columns]
            X_core = X[core_cols].values
        else:
            X_core = X[:, :6]
        X_poly = self.poly_.transform(X_core)
        return self.reg_.predict(X_poly)

    @property
    def n_bases(self):
        return self.poly_.n_output_features_


# ==============================================================================
# FIXED BASIS MODEL 2: FOURIER-FEATURE REGRESSION
# ==============================================================================
class FourierFeatureRegression(BaseEstimator, RegressorMixin):
    """
    Expands calendar time onto K fixed sinusoidal harmonic bases:
        phi_k(t) = [sin(2*pi*k*t / P), cos(2*pi*k*t / P)]
    Fitted via regularized linear regression.
    """
    def __init__(self, n_harmonics=6, period=52.1775, alpha=10.0):
        self.n_harmonics = n_harmonics
        self.period = period
        self.alpha = alpha
        self.reg_ = Ridge(alpha=alpha, random_state=42)

    def _transform_fourier(self, X):
        if isinstance(X, pd.DataFrame):
            weeks = X['week_of_year'].values
        else:
            weeks = X[:, 0]
        bases = []
        for k in range(1, self.n_harmonics + 1):
            bases.append(np.sin(2 * np.pi * k * weeks / self.period))
            bases.append(np.cos(2 * np.pi * k * weeks / self.period))
        return np.column_stack(bases)

    def fit(self, X, y):
        F = self._transform_fourier(X)
        if isinstance(X, pd.DataFrame):
            extra_cols = [c for c in ['sales_lag_1', 'Size', 'IsHoliday'] if c in X.columns]
            F = np.column_stack([F, X[extra_cols].values])
        self.reg_.fit(F, y)
        return self

    def predict(self, X):
        F = self._transform_fourier(X)
        if isinstance(X, pd.DataFrame):
            extra_cols = [c for c in ['sales_lag_1', 'Size', 'IsHoliday'] if c in X.columns]
            F = np.column_stack([F, X[extra_cols].values])
        return self.reg_.predict(F)

    @property
    def n_bases(self):
        return 2 * self.n_harmonics


# ==============================================================================
# ADAPTIVE BASIS 1: RBF NETWORK (K-MEANS LEARNED CENTERS + LINEAR HEAD)
# ==============================================================================
class RBFNetworkRegressor(BaseEstimator, RegressorMixin):
    """
Radial Basis Function (RBF) Network with learned cluster centers:
    similarity(x) = exp(-spread * distance(x, center)^2)

Where:
    - center: Cluster center found by running K-Means on the input data.
    - spread: Controls how wide or narrow each center's influence zone is.
    - weight: Importance score learned by Ridge regression to combine all similarities into the final prediction.
"""
    
    def __init__(self, n_centers=30, gamma=0.05, alpha=5.0, random_state=42):
        self.n_centers = n_centers
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state
        self.kmeans_ = None
        self.reg_ = None
        self.centers_ = None

    def _get_rbf_features(self, X):
        dist_sq = np.sum((X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        self.kmeans_ = KMeans(n_clusters=self.n_centers, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X_mat)
        self.centers_ = self.kmeans_.cluster_centers_

        H = self._get_rbf_features(X_mat)
        self.reg_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.reg_.fit(H, y)
        return self

    def predict(self, X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        H = self._get_rbf_features(X_mat)
        return self.reg_.predict(H)

    @property
    def n_bases(self):
        return self.n_centers


# ==============================================================================
# ADAPTIVE BASIS 2: MARS / SPLINETRANSFORMER ADAPTIVE SPLINES
# ==============================================================================
class MARSSplineRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_knots=6, degree=3, alpha=20.0):
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.spline_ = SplineTransformer(n_knots=n_knots, degree=degree, knots='quantile', include_bias=False)
        self.reg_ = Ridge(alpha=alpha, random_state=42)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            core_cols = [c for c in ['sales_lag_1', 'sales_roll_mean_4', 'Size', 'CPI', 'Unemployment', 'Temperature'] if c in X.columns]
            X_core = X[core_cols].values
            X_other = X[[c for c in X.columns if c not in core_cols]].values
        else:
            X_core = X[:, :6]
            X_other = X[:, 6:]

        S = self.spline_.fit_transform(X_core)
        X_comb = np.column_stack([S, X_other])
        self.reg_.fit(X_comb, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            core_cols = [c for c in ['sales_lag_1', 'sales_roll_mean_4', 'Size', 'CPI', 'Unemployment', 'Temperature'] if c in X.columns]
            X_core = X[core_cols].values
            X_other = X[[c for c in X.columns if c not in core_cols]].values
        else:
            X_core = X[:, :6]
            X_other = X[:, 6:]

        S = self.spline_.transform(X_core)
        X_comb = np.column_stack([S, X_other])
        return self.reg_.predict(X_comb)

    @property
    def n_bases(self):
        return self.spline_.n_features_out_


# ==============================================================================
# ADAPTIVE BASIS 3: GENERALIZED ADDITIVE MODEL (pygam LinearGAM)
# ==============================================================================
class PyGAMForecaster(BaseEstimator, RegressorMixin):
    """
    Generalized Additive Model:
        y = beta_0 + sum_j f_j(x_j)
    Each f_j is an adaptively penalized spline basis expansion whose coefficients
    and effective degrees of freedom are determined by penalized iteratively reweighted least squares.
    """
    def __init__(self, n_splines=10, lam=0.8):
        self.n_splines = n_splines
        self.lam = lam
        self.model_ = None

    def fit(self, X, y):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        X_sub = X_mat[:, :6]
        
        formula = s(0, n_splines=self.n_splines, lam=self.lam)
        for i in range(1, X_sub.shape[1]):
            formula += s(i, n_splines=self.n_splines, lam=self.lam)
            
        self.model_ = LinearGAM(formula)
        self.model_.fit(X_sub, y)
        return self

    def predict(self, X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        X_sub = X_mat[:, :6]
        return self.model_.predict(X_sub)

    @property
    def n_bases(self):
        return self.n_splines * 6


# ==============================================================================
# FACTORY HELPER: ALL 11 REQUIRED MODELS
# ==============================================================================
def get_all_models():
    """
    Instantiates all 11 model families conforming to the project specification.
    """
    models = {
        # A) Baselines
        'Naive (Lag 1)': NaiveForecaster(),
        'Seasonal Naive (Lag 52)': SeasonalNaiveForecaster(),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=100.0, random_state=42),
        
        # B) Fixed Basis Models
        'Polynomial Regression (deg 2)': PolynomialBasisRegression(degree=2, alpha=50.0),
        'Fourier Feature Regression': FourierFeatureRegression(n_harmonics=6, alpha=10.0),
        
        # C) Adaptive Basis Function Models
        'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
                                           max_iter=300, random_state=42, early_stopping=True),
        'RBF Network': RBFNetworkRegressor(n_centers=30, gamma=0.05, alpha=5.0, random_state=42),
        'MARS (SplineTransformer)': MARSSplineRegressor(n_knots=6, degree=3, alpha=20.0),
        'Generalized Additive Model': PyGAMForecaster(n_splines=10, lam=0.8),
        'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.85, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=150, max_depth=7, learning_rate=0.06, subsample=0.85, random_state=42, n_jobs=-1, verbose=-1)
    }
    return models
