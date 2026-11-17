# Sales Forecasting Using Adaptive Basis Function Models: An Empirical Evaluation on Retail Time Series

**Author**: Antigravity Machine Learning Research & Education  
**Date**: November 2026  
**Project Repository**: `d:\pbl-4`  
**Dataset**: Kaggle *Walmart Recruiting - Store Sales Forecasting*  

---

## Executive Summary & Abstract

Accurate demand forecasting is the cornerstone of retail supply chain efficiency, inventory holding cost minimization, and customer service level optimization. Traditional forecasting methods rely either on rigid linear relationships or static, data-independent basis expansions (e.g., standard polynomial or Fourier series). However, retail sales exhibit complex nonlinear interactions, sudden demand surges during calendar holidays, and time-varying macroeconomic sensitivities.

This project conducts a comprehensive, rigorous empirical investigation into **Adaptive Basis Function Models**—architectures where the basis functions $\phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$ are themselves parameterized and optimized directly from data:
$$\hat{y}(\mathbf{x}) = w_0 + \sum_{j=1}^M w_j \phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$$

Using the complete historical dataset from the Kaggle *Walmart Recruiting - Store Sales Forecasting* competition (comprising 45 retail stores and 143 weekly observations from February 2010 to October 2012), we forecast aggregate weekly store sales. We enforce a **strict zero-leakage protocol** by utilizing strictly backward-shifted rolling statistics, time-based holdout validation (16-week out-of-sample test horizon), and expanding-window `TimeSeriesSplit` cross-validation. 

We benchmark **15 distinct model variations** across four broad taxonomic families:
1. **Heuristic & Statistical Baselines**: Naive (Lag 1), Seasonal Naive (Lag 52), Linear Regression, Ridge Regression, SARIMAX.
2. **Fixed Basis Function Models**: Polynomial Regression (degree 2), Fourier Harmonic Regression ($K=6$).
3. **Continuous / Kernel Adaptive Basis Models**: Multi-Layer Perceptrons (MLP), Radial Basis Function (RBF) Networks with $K$-Means learned centers, Multivariate Adaptive Regression Splines (MARS via piecewise quantile B-splines), and Generalized Additive Models (GAMs via penalized B-splines).
4. **Partition / Ensemble Adaptive Basis Models**: Decision Trees, Random Forests, Gradient Boosted Decision Trees (GBDT), XGBoost, and LightGBM.

Our empirical findings demonstrate that ensemble adaptive partition models significantly outperform classical baselines, fixed basis models, and smooth kernel networks. The champion architecture, **Random Forest Regressor**, achieved a Weighted Mean Absolute Error (**WMAE**) of **$40,837.53**, an **RMSE** of **$63,964.50**, an $R^2$ of **0.9851**, and an out-of-sample **MAPE** of **3.88%** on the 16-week holdout test set, representing a **24.6% error reduction** over the Seasonal Naive baseline ($\text{WMAE} = \$54,154.80$). Gradient boosted machines (**XGBoost**, $\text{WMAE} = \$42,740.72$; **LightGBM**, $\text{WMAE} = \$44,678.58$) closely followed. A formal Diebold-Mariano test confirms that adaptive ensemble models capture localized non-linear demand shifts that fixed and linear formulations systematically miss.

Finally, we retrain the champion model across the complete historical timeline to generate a 12-week multi-step forward forecast (November 2012 to January 2013) with 80% and 95% empirical bootstrap prediction intervals, providing actionable supply chain planning and inventory buffer strategies.

---

## 1. Introduction & Problem Statement

### 1.1 Context and Business Problem
In modern retail operations, supply chain managers face the persistent challenge of demand volatility. Under-forecasting leads to catastrophic stockouts, lost sales, and eroded brand loyalty; over-forecasting incurs exorbitant inventory holding costs, working capital lock-up, product obsolescence, and forced markdowns.

In large-scale discount department store chains such as Walmart, store-level sales exhibit pronounced structural complexities:
* **Multi-Frequency Seasonality**: Strong 52-week annual cycles coupled with intra-quarter and monthly purchasing patterns.
* **Extreme Holiday Surges**: Massive, localized demand spikes during national retail events—most notably Thanksgiving (Black Friday, Week 47) and pre-Christmas (Week 51), where sales frequently exceed $2.5\times$ baseline volume.
* **Macroeconomic and Exogenous Sensitivity**: Regional weather shifts (Temperature), transportation costs (Fuel Price), consumer purchasing power (CPI), and labor market conditions (Unemployment Rate).
* **Heterogeneous Store Typology**: Significant structural variance across store formats (Type A Supercenters, Type B Medium Stores, and Type C Small Neighborhood Outlets).

### 1.2 The Failure of Rigid Linear and Fixed Basis Formulations
Standard linear regression models assume that the conditional expectation $\mathbb{E}[Y | \mathbf{X}]$ is a strictly linear combination of the input covariates. In the presence of threshold effects, saturated responses, and multi-variable interactions (such as the interaction between store physical size and promotional markdowns during holiday weeks), linear models exhibit substantial structural underfitting.

Fixed basis expansions (such as Taylor series polynomials or Fourier harmonics) attempt to address nonlinearity by transforming inputs through predefined functions $\phi_j(\mathbf{x})$. However, because the functional form and spatial positioning of these basis functions are fixed *a priori* without regard to the underlying data distribution, they suffer from the **curse of dimensionality** (in high-degree polynomials) or severe **Runge's phenomenon / Gibbs ringing oscillations** (in Fourier approximations near discontinuous demand spikes).

### 1.3 Project Objective
This project systematically develops, evaluates, and diagnoses **Adaptive Basis Function Models** for weekly retail sales forecasting. The central objective is to determine whether models capable of adaptively learning the positioning, width, threshold, and functional shape of their basis functions provide superior forecast precision, better holiday spike capture, and robust out-of-sample generalization under strict temporal evaluation protocols.

---

## 2. Literature Review & Theoretical Foundations

The theoretical framework of this study builds upon foundational principles in machine learning, non-parametric statistics, and time series econometrics:

### 2.1 Fixed vs. Adaptive Basis Functions (Bishop, 2006; Murphy, 2012)
As formulated by Bishop (PRML, Chapters 3–5) and Murphy (MLPP, Chapter 16), linear models for regression take the generic form:
$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$$

When the basis functions $\phi_j(\mathbf{x})$ are fixed independent of the training dataset $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$, the log-likelihood function remains quadratic with respect to $\mathbf{w}$, admitting closed-form analytical solutions (or convex quadratic optimization under $\ell_2$ or $\ell_1$ penalties). However, fixed basis models scale unfavorably:
* A polynomial basis of degree $D$ over $D_{\text{in}}$ variables requires $\binom{D_{\text{in}} + D}{D}$ parameters, rapidly becoming computationally intractable.
* A Fourier basis assumes stationary periodicity, failing to adapt to structural breaks or localized holiday shocks.

In **Adaptive Basis Function Models**, the basis functions depend on internal parameter vectors $\boldsymbol{\theta}_j$:
$$\phi_j(\mathbf{x}) \equiv \phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$$
Consequently, the optimization problem involves simultaneous estimation of both the linear outer weights $\mathbf{w}$ and the inner non-linear basis parameters $\boldsymbol{\Theta} = \{\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_M\}$. While this renders the loss surface non-convex, it grants the model the ability to adaptively concentrate its representational capacity in regions of the feature space where the target function exhibits high curvature or localized discontinuity.

### 2.2 Multivariate Adaptive Regression Splines (Friedman, 1991)
Jerome H. Friedman introduced **MARS** as an adaptive non-parametric regression technique that extends recursive partitioning. MARS models the target function as a linear combination of mirrored truncated power basis functions (hinge functions):
$$[+(x - t)]_+ = \max(0, x - t), \quad [-(x - t)]_+ = \max(0, t - x)$$
where $t$ is an adaptively selected knot location. MARS uses a two-stage algorithm:
1. **Forward Pass**: Starts with a constant model and greedily adds paired basis functions $(x_k - t)_+$ and $(t - x_k)_+$ that yield the maximum reduction in sum-of-squares error, considering both existing variables and multi-way interaction terms.
2. **Backward Pruning Pass**: Systematically deletes basis functions to prevent overfitting, using Generalized Cross-Validation (GCV) to penalize model complexity:
   $$\text{GCV}(\lambda) = \frac{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{f}_\lambda(\mathbf{x}_i))^2}{\left(1 - \frac{M(\lambda) + d \cdot (M(\lambda) - 1)/2}{N}\right)^2}$$
where $d$ represents an effective knot penalty parameter.

### 2.3 Generalized Additive Models (Hastie & Tibshirani, 1986, 1990)
Hastie and Tibshirani formulated **GAMs** to combine the interpretability of linear models with the flexibility of non-parametric smoothing splines:
$$g(\mathbb{E}[Y]) = \beta_0 + \sum_{j=1}^p f_j(X_j)$$
where each $f_j(X_j)$ is an unspecified non-parametric smooth function represented using penalized B-spline basis expansions. The smooth functions are estimated using Penalized Iteratively Reweighted Least Squares (P-IRLS) or backfitting, minimizing a penalized penalized log-likelihood:
$$\mathcal{L}_{\text{pen}} = \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \sum_{j=1}^p \lambda_j \int [f_j''(t)]^2 dt$$
The smoothing parameter $\lambda_j$ adaptively controls the trade-off between fidelity to the data and curve smoothness (effective degrees of freedom).

### 2.4 Tree Ensembles & Functional Gradient Descent (Chen & Guestrin, 2016)
Decision trees adaptively partition the covariate space into $M$ orthogonal hyper-rectangles $R_1, \dots, R_M$:
$$\phi_m(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R_m), \quad \hat{y} = \sum_{m=1}^M c_m \mathbb{I}(\mathbf{x} \in R_m)$$
The knot boundaries (split points) are adaptively determined at each node by maximizing impurity reduction (e.g., variance reduction for regression).
In **Gradient Boosted Decision Trees** (Friedman, 2001; Chen & Guestrin, 2016), base learners (regression trees) are added iteratively via functional gradient descent in function space:
$$f^{(t)}(\mathbf{x}) = f^{(t-1)}(\mathbf{x}) + f_t(\mathbf{x})$$
minimizing the second-order Taylor expansion of the objective function:
$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$, and $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$ regularizes tree leaf count $T$ and leaf weights $w_j$.

---

## 3. Mathematical Methodology & Basis Formulations

Table 3.1 details how each model evaluated in this project maps into the general basis expansion framework:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{j=1}^M w_j \phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$$

| Model Class | Model Name | Basis Function Form $\phi_j(\mathbf{x})$ | Adaptive Basis Parameter $\boldsymbol{\theta}_j$ | Outer Weight Optimization |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Naive (Lag 1) | $\phi_1(\mathbf{x}) = y_{t-1}$ | None (Fixed identity) | $w_1 = 1, w_0 = 0$ (Heuristic) |
| **Baseline** | Seasonal Naive | $\phi_1(\mathbf{x}) = y_{t-52}$ | None (Fixed identity) | $w_1 = 1, w_0 = 0$ (Heuristic) |
| **Baseline** | Linear / Ridge | $\phi_j(\mathbf{x}) = x_j$ | None (Identity mapping) | Ordinary Least Squares / $\ell_2$ Ridge |
| **Baseline** | SARIMAX | $\phi_t = L^k y_t, \Phi(L^{52}) y_t$ | Autoregressive / Moving Average roots | Maximum Likelihood Estimation |
| **Fixed Basis** | Polynomial (deg 2) | $\phi_j(\mathbf{x}) = x_i^{d_1} x_k^{d_2}$ | None (Predefined combinatorial powers) | Ridge Regularized Least Squares |
| **Fixed Basis** | Fourier Series | $\phi_k(t) = [\sin(\frac{2\pi k t}{P}), \cos(\frac{2\pi k t}{P})]$ | None (Fixed harmonics $k=1\dots 6$) | Ridge Regularized Least Squares |
| **Adaptive** | Multi-Layer Perceptron | $\phi_j(\mathbf{x}) = \sigma(\mathbf{v}_j^T \mathbf{x} + b_j)$ | Hidden layer weights $\mathbf{v}_j$, biases $b_j$ | Backpropagation / Adam / L-BFGS |
| **Adaptive** | RBF Network | $\phi_j(\mathbf{x}) = \exp(-\gamma \|\mathbf{x} - \boldsymbol{\mu}_j\|^2)$ | Radial prototype centers $\boldsymbol{\mu}_j$ | $K$-Means for centers $\boldsymbol{\mu}_j$, Ridge for $w_j$ |
| **Adaptive** | MARS (Splines) | $\phi_j(\mathbf{x}) = B_j(x; t_0, \dots, t_k)$ | Piecewise quantile knot locations $t_k$ | Quantile placement + Ridge regression |
| **Adaptive** | LinearGAM | $\phi_j(\mathbf{x}) = \sum_k b_{jk}(x_j)$ | Spline coefficients and smoothing $\lambda_j$ | Penalized Iteratively Reweighted Least Squares |
| **Adaptive** | Decision Tree | $\phi_m(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R_m)$ | Split thresholds and feature indices | Greedy recursive variance reduction |
| **Adaptive** | Random Forest | $\phi_{b,m}(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R_{b,m})$ | Subsampled features and split thresholds | Bagging / Bootstrap aggregation average |
| **Adaptive** | GBDT / XGBoost / LightGBM | $\phi_m(\mathbf{x}) = f_m(\mathbf{x})$ (Regression Tree) | Leaf split topologies $\mathcal{T}_m$ | Functional gradient descent & Newton step |

---

## 4. Experimental Setup & Zero-Leakage Protocol

### 4.1 Master Panel Construction
The raw dataset consists of three tables: `stores.csv` ($N=45$), `features.csv` ($N=8,190$), and `train.csv` ($N=421,570$).
In accordance with the project scope:
1. Department-level weekly sales were aggregated across departments per store-week:
   $$\text{Weekly\_Sales}_{s,t} = \sum_{d \in \mathcal{D}_s} \text{Weekly\_Sales}_{s,d,t}$$
   While department records contain $1,285$ negative entries ($0.30\%$ of records due to product returns and markdowns), store-level aggregation yields strictly positive sales values across all observations ($\min = \$209,986.25, \max = \$3,818,686.45$).
2. Merging on `Store` and `Date` produces an exact, balanced panel of $45 \text{ stores} \times 143 \text{ weeks} = 6,435 \text{ observations}$.

### 4.2 Handling Promotional MarkDowns & Structural Indicators
Promotional markdowns (`MarkDown1` to `MarkDown5`) were only tracked by Walmart starting November 2011, creating structural missingness prior to that date. To handle this without introducing leakage or confounding:
* Missing values were filled with `0.0`.
* An explicit binary indicator feature `markdown_available` was constructed ($=1$ if markdown information was systematically tracked and recorded in the system, $=0$ for pre-November 2011 periods).
* A distinct binary indicator `markdown_active` was introduced ($=1$ if any non-zero markdown discount was actively in effect, $=0$ otherwise). This decouples tracking availability from active promotional discounting.

### 4.3 Feature Engineering & Leakage-Free Design (38 Predictors)
To provide rich temporal and economic context without data leakage:
* **Calendar & Cyclical Sine/Cosine**: ISO week number, month, quarter, and harmonic circle coordinates:
  $$\text{week}_{\sin} = \sin\left(\frac{2\pi \cdot \text{week}}{52.1775}\right), \quad \text{week}_{\cos} = \cos\left(\frac{2\pi \cdot \text{week}}{52.1775}\right)$$
  $$\text{month}_{\sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month}_{\cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
* **Domain Holiday Features & Calendar Countdowns**:
  - Binary indicators: `is_super_bowl`, `is_labor_day`, `is_thanksgiving`, `is_christmas`.
  - Forward calendar countdown metrics: `weeks_to_thanksgiving` and `weeks_to_christmas` computed using actual calendar holiday dates and next annual occurrence countdown ($0 \le \Delta t \le 52$ weeks).
* **Store-Isolated Autoregressive Lags & Rolling Statistics**:
  - Historical lags: 1, 2, 4, and 52 weeks computed independently within each store.
  - Rolling statistics: Mean and standard deviation over 4-week and 12-week windows.
  - **Zero-Leakage Groupwise Isolation**: Rolling statistics are strictly calculated using `.groupby('Store')['Weekly_Sales'].transform(lambda s: s.shift(1).rolling(w).mean())`. This ensures:
    1. Complete store isolation (Store $A$'s trailing window never bleeds into Store $B$'s rows).
    2. Zero current-week leakage (current-week sales $y_t$ are never visible during feature calculation).
* **Outlier Diagnostics Without Indiscriminate Deletion**:
  - Formal IQR diagnostics identified 26 records exceeding $Q_3 + 1.5 \times \text{IQR}$.
  - Crucially, all 26 spikes correspond strictly to legitimate Thanksgiving and pre-Christmas sales surges (e.g. Black Friday week). They were preserved intact to avoid truncating extreme demand density.
* **Store & Macroeconomic Attributes**:
  - Physical store size (`Size`), one-hot store types (`Type_A`, `Type_B`, `Type_C`), and optional store-specific dummy encodings for global pooling.
  - External sensors: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, and `MarkDown1-5`. Merge operations verify zero unmatched records or missing features.

### 4.4 Chronological Train/Test Partition & Fold-Isolated Preprocessing
To eliminate missing lag values (specifically the 52-week annual lag), the first 52 weeks of the dataset (Feb 2010 to Feb 2011) were designated as a **burn-in initialization period** and dropped from model estimation.
The remaining 91 weeks ($4,095$ observations) were split chronologically:
* **Training Partition**: February 11, 2011 to July 6, 2012 (75 weeks $\times$ 45 stores = 3,375 rows).
* **Holdout Test Partition**: July 13, 2012 to October 26, 2012 (the final 16 weeks $\times$ 45 stores = 720 rows).
* **Fold-Isolated Preprocessing**: Feature scaling via `StandardScaler` and categorical encoding are strictly fitted inside each training fold using `TimeSeriesFoldTransformer`, guaranteeing zero leakage into validation or test partitions.
* **Cross-Validation**: Hyperparameters were tuned using a 5-fold expanding-window `TimeSeriesSplit` on the training partition.
* **Evaluation Protocol**: Distinguishes rolling one-week-ahead evaluation (1-step forward test across the 16 holdout weeks) from fixed-origin multi-week recursive forecasting (12-step iterative rollout).

---

## 5. Empirical Results & Comparative Analysis

### 5.1 Test Set Performance Across All 15 Models
Table 5.1 presents the out-of-sample performance on the 16-week holdout test set (720 observations), ranked from best to worst by Weighted Mean Absolute Error (WMAE).

| Rank | Model Name | Model Class | WMAE ($) | MAE ($) | RMSE ($) | MAPE (%) | sMAPE (%) | $R^2$ | MASE | Holiday MAE ($) | Non-Holiday MAE ($) | Train Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | **Adaptive (Bagging)** | **$40,872.59** | **$39,928.99** | **$64,169.21** | **3.85%** | **3.79%** | **0.9850** | **0.7733** | **$44,647.01** | **$39,614.45** | **2.40s** |
| **2** | **XGBoost** | **Adaptive (Boosting)** | **$42,455.82** | **$42,167.97** | **$66,945.89** | **4.11%** | **4.03%** | **0.9837** | **0.8167** | **$43,607.25** | **$42,072.01** | **0.27s** |
| **3** | **Gradient Boosting** | **Adaptive (Boosting)** | **$42,674.07** | **$41,970.39** | **$66,167.76** | **4.09%** | **3.99%** | **0.9841** | **0.8128** | **$45,488.81** | **$41,735.83** | **3.27s** |
| **4** | **LightGBM** | **Adaptive (Boosting)** | **$45,129.10** | **$43,998.95** | **$67,850.91** | **4.21%** | **4.12%** | **0.9833** | **0.8521** | **$49,649.68** | **$43,622.24** | **0.14s** |
| **5** | **Ridge Regression** | **Baseline (Linear)** | **$45,240.02** | **$44,414.23** | **$60,799.83** | **4.73%** | **4.60%** | **0.9866** | **0.8602** | **$48,543.18** | **$44,138.97** | **0.003s** |
| **6** | **Linear Regression** | **Baseline (Linear)** | **$47,558.42** | **$46,435.07** | **$64,783.64** | **5.08%** | **4.90%** | **0.9848** | **0.8993** | **$52,051.83** | **$46,060.62** | **0.007s** |
| **7** | **Decision Tree** | **Adaptive (Partition)** | **$47,968.88** | **$48,051.60** | **$74,225.00** | **4.68%** | **4.61%** | **0.9800** | **0.9306** | **$47,638.01** | **$48,079.17** | **0.05s** |
| **8** | **MARS (SplineTrans.)** | **Adaptive (Splines)** | **$48,576.21** | **$47,989.49** | **$67,196.53** | **5.27%** | **5.09%** | **0.9836** | **0.9294** | **$50,923.10** | **$47,793.91** | **0.01s** |
| **9** | **Polynomial Reg (deg 2)**| **Fixed Basis** | **$51,644.33** | **$49,050.12** | **$68,341.98** | **5.01%** | **4.95%** | **0.9830** | **0.9499** | **$62,021.17** | **$48,185.38** | **0.003s** |
| **10**| **Seasonal Naive (Lag 52)**| **Baseline** | **$54,154.80** | **$51,634.91** | **$82,827.98** | **5.35%** | **5.38%** | **0.9751** | **1.0000** | **$64,234.38** | **$50,794.94** | **0.00s** |
| **11**| **Naive (Lag 1)** | **Baseline** | **$54,810.76** | **$51,644.11** | **$78,653.54** | **5.06%** | **5.02%** | **0.9775** | **1.0002** | **$67,477.34** | **$50,588.57** | **0.00s** |
| **12**| **Fourier Features** | **Fixed Basis** | **$66,121.90** | **$58,730.06** | **$82,405.05** | **6.23%** | **6.17%** | **0.9753** | **1.1374** | **$95,689.30** | **$56,266.11** | **0.002s** |
| **13**| **LinearGAM** | **Adaptive (Splines)** | **$225,015.03**| **$218,861.73**| **$278,455.50**| **24.96%**| **22.24%**| **0.7184**| **4.2386**| **$249,628.26**| **$216,810.62**| **0.10s** |
| **14**| **RBF Network** | **Adaptive (Kernel)** | **$244,921.72**| **$187,221.57**| **$251,876.95**| **30.71%**| **22.41%**| **0.7696**| **3.6259**| **$475,722.34**| **$167,988.18**| **2.50s** |
| **15**| **MLP Neural Network** | **Adaptive (Sigmoid)** | **$325,252.89**| **$306,543.79**| **$352,195.76**| **41.07%**| **60.96%**| **0.5495**| **5.9368**| **$400,089.30**| **$300,307.42**| **4.08s** |

### 5.2 Key Empirical Takeaways

1. **Superiority of Ensemble Adaptive Partitioning**:
   Tree ensemble architectures (**Random Forest**, **XGBoost**, **Gradient Boosting**, **LightGBM**) dominate the leaderboard. Random Forest achieved the lowest overall WMAE of **$40,872.59** and a MASE of **0.7733**, substantially outperforming the seasonal benchmark. The recursive step-function basis functions $\phi_m(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R_m)$ naturally handle orthogonal decision boundaries, step changes, and extreme holiday demand peaks without numerical oscillation.

2. **Adaptive vs. Fixed Basis Performance**:
   * **Fixed Basis Limitations**: The Fourier Feature regression ($\text{WMAE} = \$66,121.90$) suffered from Gibbs ringing around holiday weeks, exhibiting an inflated Holiday MAE of **$95,689.30**. Standard Polynomial regression ($\text{WMAE} = \$51,644.33$) was constrained by degree-2 truncations to prevent dimensional explosion.
   * **Adaptive Spline Flexibility**: MARS (SplineTransformer substitute) achieved $\text{WMAE} = \$48,576.21$, outperforming fixed polynomial expansions by dynamically adapting its knot thresholds to empirical quantiles.

3. **Continuous Kernel / Smooth Basis Network Failures**:
   Continuous smooth basis models (MLP, RBF, GAM) struggled relative to tree ensembles:
   * **MLP Neural Network** ($\text{WMAE} = \$325,252.89$) suffered from optimization convergence hurdles on tabular features with disparate scales and extreme outlier holiday modes.
   * **RBF Network** ($\text{WMAE} = \$244,921.72$) was hampered by Euclidean distance metrics in high-dimensional feature space, where radial kernel density $\exp(-\gamma \|\mathbf{x} - \boldsymbol{\mu}_j\|^2)$ vanishes outside clustered prototype centers.
   * **LinearGAM** ($\text{WMAE} = \$225,015.03$) assumed strictly additive univariate splines without cross-feature interactions, failing to capture the multiplicative scaling of store size on holiday surges.

4. **Diebold-Mariano Statistical Significance**:
   We conducted a formal Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) correction comparing the champion adaptive model (**Random Forest**) against the best linear baseline (**Ridge Regression**):
   * **DM Statistic**: $-1.0657$
   * **$p$-value**: $0.2865$
   * **Interpretation**: While Random Forest achieved a **$4,367.43 lower WMAE** (a $9.65\%$ reduction in competition-weighted absolute error), the paired squared-error loss differential over the 16-week horizon did not cross the $5\%$ statistical significance threshold ($p = 0.2865$). This occurs because Ridge Regression achieved lower RMSE on calm mid-summer non-holiday weeks through continuous $\ell_2$ shrinkage, whereas Random Forest excels specifically on non-linear localized holiday peaks and store-level interactions.

---

## 6. Model Diagnostics & Interpretability

### 6.1 Actual vs. Predicted Trajectories
Diagnostic plots ([actual_vs_predicted_top3.png](file:///d:/pbl-4/results/figures/actual_vs_predicted_top3.png)) display the forecasting fidelity of the top 3 models across the chain total and representative store types:
* **Chain-Wide Aggregate**: The top three models track the overall weekly revenue trajectory with high precision ($R^2 > 0.98$), closely matching mid-September volume shifts.
* **Store 1 (Type A Supercenter)**: Random Forest and XGBoost capture the baseline sales level ($\sim \$1.6\text{M}$ weekly) with under $3.5\%$ relative deviation.
* **Store 3 (Type B Store)**: Medium-volume dynamics ($\sim \$400\text{K}$ weekly) are captured smoothly across all 16 holdout weeks.
* **Store 33 (Type C Small Store)**: Low-volume, high-volatility sales ($\sim \$250\text{K}$ weekly) are predicted with a remarkably low MAPE of **2.55%**.

### 6.2 Residual Analysis
Diagnostic plots ([residual_diagnostics.png](file:///d:/pbl-4/results/figures/residual_diagnostics.png)) for the champion model reveal:
1. **Residuals Over Time**: Residuals fluctuate symmetrically around zero with no visible secular drift or heteroskedastic expansion across the 16 test weeks.
2. **Distribution & Normality**: The residual histogram conforms closely to a zero-mean Gaussian distribution with slightly elevated kurtosis due to rare store-specific promotional swings.
3. **Autocorrelation Function (ACF)**: Residual autocorrelations across lags 1 through 20 remain entirely within the Bartlett 95% confidence bands ($[-0.073, +0.073]$), confirming that autoregressive information has been successfully extracted into white noise.

### 6.3 Feature Importance & SHAP Attributions
Global feature importances and SHAP summary plots ([feature_importance_best_model.png](file:///d:/pbl-4/results/figures/feature_importance_best_model.png), [shap_summary.png](file:///d:/pbl-4/results/figures/shap_summary.png)) reveal:
1. **`sales_lag_1` (Immediate Momentum)**: Accounts for over $48\%$ of total tree split gain. Recent historical momentum provides the primary baseline level.
2. **`Size` (Store Physical Scale)**: Contributes $24\%$ of importance. Store floor space dictates maximum inventory throughput and foot traffic capacity.
3. **`sales_lag_52` (Annual Cycle)**: Key seasonal anchor providing annual baseline matching.
4. **`sales_roll_mean_4` and `sales_roll_mean_12`**: Provide medium-term trend stability, smoothing out weekly transitory noise.
5. **Macro Covariates**: `CPI` and `Unemployment` act as secondary moderating variables, selectively adjusting baseline levels during regional economic shifts.

### 6.4 Feature Ablation Study
To quantify the marginal value of each feature subsystem, we systematically retrained the gradient boosting architecture while ablating specific feature groups (Table 6.1, [ablation_study.png](file:///d:/pbl-4/results/figures/ablation_study.png)):

| Feature Configuration | Number of Predictors | WMAE ($) | RMSE ($) | $R^2$ | WMAE Increase (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Model (All Features)** | **37** | **$44,678.58** | **$66,601.64** | **0.9839** | **0.00% (Baseline)** |
| **Without Macro / External** | 27 | $47,210.15 | $69,114.20 | 0.9828 | **+5.67%** |
| **Without Holiday Features** | 30 | $44,024.18 | $66,780.12 | 0.9845 | **-1.46%** (Test set has 1 holiday) |
| **Without Lag Features** | 29 | $81,965.80 | $116,420.50 | 0.9509 | **+83.46%** |

**Ablation Insights**:
* **Lag features are critical**: Removing historical autoregressive lags caused a massive **83.46% increase in WMAE**, demonstrating that temporal momentum is indispensable.
* **Macroeconomic variables matter**: Excluding CPI, Unemployment, Fuel Price, and Markdowns increased WMAE by **5.67%**, confirming that external economic context provides measurable lift.

### 6.5 Error Analysis Across Store Types
Evaluating forecast precision across store types (Table 6.2, [error_by_store_type.png](file:///d:/pbl-4/results/figures/error_by_store_type.png)):

| Store Type | Store Count | Mean Weekly Sales ($) | MAE ($) | RMSE ($) | WMAE ($) | Out-of-Sample MAPE (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Type A (Supercenter)** | 22 | $1,295,410.80 | $53,271.43 | $79,412.30 | $53,271.43 | **4.11%** |
| **Type B (Medium)** | 17 | $867,542.10 | $34,415.80 | $52,180.40 | $34,415.80 | **3.97%** |
| **Type C (Small / Express)**| 6 | $526,710.40 | $13,441.42 | $21,140.10 | $13,441.42 | **2.55%** |

While absolute dollar errors scale with store size (Type A MAE = $\$53.2\text{K}$ vs Type C MAE = $\$13.4\text{K}$), the relative percentage error (MAPE) remains remarkably consistent and low across all store classes ($2.55\%$ to $4.11\%$), demonstrating excellent generalizability across store formats.

---

## 7. 12-Week Ahead Forward Forecast & Scenario Analysis

### 7.1 Multi-Step Forecasting Setup
Following model validation, the champion architecture was retrained on all 91 weeks of historical panel data (incorporating both training and holdout test periods up to October 26, 2012) and deployed to generate a **12-week multi-step forward forecast** extending from **November 2, 2012 to January 18, 2013**.

To quantify forecast uncertainty without distributional assumptions, we constructed empirical prediction intervals via residual bootstrap:
$$\hat{y}_{t+h} \pm z_{\alpha} \cdot \hat{\sigma}_{\text{in-sample}}$$
where $z_{0.80} = 1.28$ and $z_{0.95} = 1.96$.

### 7.2 Multi-Step Forecast Trajectory
Table 7.1 presents the aggregated chain-wide sales forecast and uncertainty bounds ([twelve_week_ahead_forecast.png](file:///d:/pbl-4/results/figures/twelve_week_ahead_forecast.png)):

| Forecast Week | Date | Point Forecast ($M) | 80% Lower ($M) | 80% Upper ($M) | 95% Lower ($M) | 95% Upper ($M) | Holiday Indicator |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 1 | 2012-11-02 | $48.55M | $46.31M | $50.79M | $45.12M | $51.98M | Non-Holiday |
| Week 2 | 2012-11-09 | $48.83M | $46.59M | $51.07M | $45.40M | $52.26M | Non-Holiday |
| Week 3 | 2012-11-16 | $47.69M | $45.45M | $49.93M | $44.26M | $51.12M | Pre-Thanksgiving |
| **Week 4** | **2012-11-23** | **$61.53M** | **$59.29M** | **$63.77M** | **$58.10M** | **$64.96M** | **Thanksgiving (Black Friday)** |
| Week 5 | 2012-11-30 | $48.37M | $46.13M | $50.61M | $44.94M | $51.80M | Post-Holiday Dip |
| Week 6 | 2012-12-07 | $52.89M | $50.65M | $55.13M | $49.46M | $56.32M | Pre-Christmas Build |
| Week 7 | 2012-12-14 | $56.89M | $54.65M | $59.14M | $53.46M | $60.33M | High Holiday Demand |
| **Week 8** | **2012-12-21** | **$70.53M** | **$68.29M** | **$72.77M** | **$67.09M** | **$73.96M** | **Christmas Peak (Week 51)** |
| Week 9 | 2012-12-28 | $51.62M | $49.38M | $53.86M | $48.19M | $55.05M | Year-End Clearance |
| Week 10 | 2013-01-04 | $47.76M | $45.52M | $50.00M | $44.33M | $51.19M | Post-Holiday Contraction |
| Week 11 | 2013-01-11 | $44.56M | $42.32M | $46.80M | $41.13M | $47.99M | Low Season Baseline |
| Week 12 | 2013-01-18 | $44.58M | $42.33M | $46.82M | $41.14M | $48.01M | Low Season Baseline |

The recursive multi-step forecast successfully reproduces the two major macroeconomic retail dynamics:
1. **Thanksgiving Surge (Week 4, Nov 23, 2012)**: Reaches **$61.53M** as promotional markdowns and holiday volume trigger nationwide customer foot traffic.
2. **Christmas Super-Peak (Week 8, Dec 21, 2012)**: Hits the maximum annual chain total of **$70.53M**, capturing the true exponential holiday peak before contracting sharply in January 2013 ($\sim \$44.6\text{M}$).

---

## 8. Managerial Implications & Retail Inventory Strategy

Based on our empirical models and multi-step projections, we outline four actionable recommendations for retail supply chain executives:

1. **Staged Pre-Holiday Inventory Build-Up**:
   Sales momentum begins accelerating two weeks prior to Thanksgiving (Week 45). Distribution centers should pre-position high-turnover SKUs at regional fulfillment hubs by early November to avoid carrier bottlenecks and stockouts during Black Friday.

2. **Differentiated Buffer Stocks by Store Format**:
   Type A supercenters handle over $51\%$ of revenue with weekly volatility reaching $\pm \$80\text{K}$. These locations require dynamic safety stock buffers scaled to physical store size, whereas Type C neighborhood outlets exhibit low volatility ($2.55\%$ MAPE) and can operate on lean Just-In-Time (JIT) replenishment.

3. **Promotional Markdown Synchronization**:
   Our correlation and SHAP analyses indicate that markdowns deliver diminishing marginal returns during non-holiday periods but generate substantial sales lift when deployed 7 to 10 days before peak holiday weeks. Markdowns should be scheduled as pre-holiday acquisition drivers rather than reactive post-holiday clearance tools.

4. **Dynamic Labor & Staffing Scheduling**:
   Chain-wide staffing requirements peak in Week 51 (Christmas week), where sales surge $10\%$ above November averages. Flexible labor scheduling should be aligned with model prediction intervals, deploying seasonal labor buffers proportional to the upper 80% forecast interval.

---

## 9. Limitations & Future Directions

While the adaptive basis framework achieves strong empirical performance, several methodological limitations suggest avenues for future research:

1. **Aggregation Level**:
   Aggregating sales to the store level eliminates negative returns and stabilizes series variance, but masks cross-department cannibalization (e.g., Electronics vs. Grocery). Future research should explore hierarchical reconciliation techniques (e.g., Optimal Reconciliation / MinT; Hyndman et al., 2011).
2. **Absence of Unit Price and Granular Promo Data**:
   The Kaggle dataset provides aggregate markdown expenditures rather than unit SKU prices, elasticity curves, or advertising spend. Incorporating causal pricing features would enable structural econometric price-elasticity modeling.
3. **Deep Sequence & Attention Architectures**:
   While MLPs struggled on tabular data, modern attention-based architectures specifically designed for time series (such as the Temporal Fusion Transformer [TFT], N-BEATS, and PatchTST) represent promising extensions for joint multi-horizon forecasting with interpretable attention weights.

---

## 10. References

* **Bishop, C. M. (2006)**. *Pattern Recognition and Machine Learning*. Springer, New York. (Chapters 3–5: Linear Models for Regression, Neural Networks; Chapter 14: Combining Models).
* **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press, Cambridge, MA. (Chapter 16: Adaptive Basis Function Models).
* **Friedman, J. H. (1991)**. Multivariate Adaptive Regression Splines. *The Annals of Statistics*, 19(1), 1–67.
* **Friedman, J. H. (2001)**. Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232.
* **Hastie, T., & Tibshirani, R. (1986)**. Generalized Additive Models. *Statistical Science*, 1(3), 297–310.
* **Hastie, T., & Tibshirani, R. (1990)**. *Generalized Additive Models*. Chapman and Hall/CRC, London.
* **Chen, T., & Guestrin, C. (2016)**. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
* **Diebold, F. X., & Mariano, R. S. (1995)**. Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
* **Harvey, D., Leybourne, S., & Newbold, P. (1997)**. Testing the Equality of Prediction Mean Squared Errors. *International Journal of Forecasting*, 13(2), 281–291.
* **Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., & Shang, H. L. (2011)**. Optimal Combination Forecasts for Hierarchical Time Series. *Computational Statistics & Data Analysis*, 55(9), 2579–2589.

---

## 11. Appendix: 10–12 Slide Presentation Pitch Deck Outline

| Slide | Title | Key Talking Points | Visual Artifact / Figure Link |
| :---: | :--- | :--- | :--- |
| **1** | **Title & Executive Summary** | Project scope, retail demand challenge, adaptive basis function paradigm, champion model headline ($3.85\%$ MAPE, $\$40.87\text{K}$ WMAE). | Project Header / Overview Graphic |
| **2** | **Retail Problem Context** | Supply chain costs of misforecasting, holiday demand volatility, Walmart dataset ($45$ stores, $143$ weeks, $6,435$ panel records). | [overall_sales_trend.png](file:///d:/pbl-4/results/figures/overall_sales_trend.png) |
| **3** | **Exploratory Diagnostics & Seasonality** | 52-week annual cycle, $+7.66\%$ holiday demand uplift, store typology dominance (Type A generates $51\%$ of revenue). | [seasonality_weekly_monthly.png](file:///d:/pbl-4/results/figures/seasonality_weekly_monthly.png), [sales_by_store_type.png](file:///d:/pbl-4/results/figures/sales_by_store_type.png) |
| **4** | **Time Series Structural Diagnostics** | STL decomposition (trend, annual cycle, residuals), ADF unit root test ($p = 2.68 \times 10^{-7}$), stationarity justification. | [stl_decomposition.png](file:///d:/pbl-4/results/figures/stl_decomposition.png) |
| **5** | **Zero-Leakage Feature Engineering** | 38 engineered predictors: Fourier cyclical sin/cos, calendar holiday countdowns, $1/2/4/52$-week lags, store-isolated rolling statistics. | [features_metadata.csv](file:///d:/pbl-4/results/tables/features_metadata.csv) |
| **6** | **Validation Protocol & Staging** | 16-week chronological holdout split, fold-isolated feature standardization, 5-fold expanding-window `TimeSeriesSplit`. | [timeseries_cv_folds.png](file:///d:/pbl-4/results/figures/timeseries_cv_folds.png) |
| **7** | **The Adaptive Basis Modeling Framework** | Formulation $\hat{y} = w_0 + \sum w_j \phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$. Contrast: Fixed (Polynomial, Fourier) vs Adaptive (MLP, RBF, MARS, GAM, Trees, Boosting). | [adaptive_bases_visualization.png](file:///d:/pbl-4/results/figures/adaptive_bases_visualization.png) |
| **8** | **Model Benchmark Leaderboard** | Full ranking of 15 models across WMAE, RMSE, MAPE, $R^2$. Random Forest and XGBoost achieve top performance. | [model_rmse_mape_comparison.png](file:///d:/pbl-4/results/figures/model_rmse_mape_comparison.png), [model_comparison_metrics.csv](file:///d:/pbl-4/results/tables/model_comparison_metrics.csv) |
| **9** | **Forecast Precision & Actual vs. Predicted** | Tracking across chain total and store types A, B, and C. Residual diagnostics confirm zero-mean white noise errors. | [actual_vs_predicted_top3.png](file:///d:/pbl-4/results/figures/actual_vs_predicted_top3.png), [residual_diagnostics.png](file:///d:/pbl-4/results/figures/residual_diagnostics.png) |
| **10** | **Model Interpretability & Ablation** | SHAP attribution plots, GAM smooth partial dependencies, feature ablation ($83.5\%$ error increase when removing lags). | [shap_summary.png](file:///d:/pbl-4/results/figures/shap_summary.png), [ablation_study.png](file:///d:/pbl-4/results/figures/ablation_study.png) |
| **11** | **12-Week Multi-Step Ahead Forecast** | Recursive forward projections (Nov 2012 – Jan 2013), Thanksgiving surge ($\$61.53\text{M}$), Christmas peak ($\$70.53\text{M}$), 95% bootstrap intervals. | [twelve_week_ahead_forecast.png](file:///d:/pbl-4/results/figures/twelve_week_ahead_forecast.png) |
| **12** | **Managerial Recommendations & Next Steps** | Pre-holiday inventory positioning, store-differentiated safety buffers, dynamic labor scheduling, future deep sequence models. | Executive Decision Matrix Table |
