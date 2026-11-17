# Sales Forecasting using Adaptive Basis Function Models

An end-to-end, reproducible machine learning research project and educational benchmark evaluating **adaptive basis function models** against fixed-basis and linear baselines for multi-store retail sales forecasting.

Benchmarked on the Kaggle *Walmart Recruiting - Store Sales Forecasting* dataset (45 stores, 143 weekly observations, February 2010 to October 2012).

---

## 📌 Problem Statement

Retail businesses require accurate, robust sales forecasts to optimize inventory, labor scheduling, and supply chain logistics. Retail demand exhibits sharp non-linearities: seasonal oscillations, extreme holiday volume surges (Thanksgiving/Black Friday, Christmas), and store-level heterogeneity.

Traditional linear or fixed polynomial/Fourier models struggle with non-local step changes and complex interactions. This project formulates and evaluates **adaptive basis function models**—where basis functions $\phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$ are learned adaptively from data rather than fixed a priori:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{j=1}^M w_j \phi_j(\mathbf{x}; \boldsymbol{\theta}_j)$$

---

## 🧠 Evaluated Model Families (15 Models)

### A. Baselines
1. **Naive (Lag 1)**: Last observed weekly sales.
2. **Seasonal Naive (Lag 52)**: Sales observed 52 weeks ago.
3. **Linear Regression (OLS)**: Ordinary least squares.
4. **Ridge Regression ($\ell_2$)**: Continuous regularized linear baseline.

### B. Fixed Basis Function Models
5. **Polynomial Regression (Degree 2)**: Combinatorial fixed polynomial expansions.
6. **Fourier Feature Regression**: Fixed harmonic sine/cosine seasonal basis functions.

### C. Adaptive Basis Function Models (Core)
7. **Decision Tree**: Adaptive axis-aligned partition step functions $\phi_m(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R_m)$.
8. **Random Forest**: Bagged ensembles of decorrelated adaptive partition trees.
9. **Gradient Boosted Decision Trees (GBDT)**: Sequentially fitted shallow regression trees minimizing pseudo-residuals.
10. **XGBoost**: Extreme Gradient Boosting with exact greedy split finding and second-order Taylor expansion.
11. **LightGBM**: Fast leaf-wise (best-first) tree expansion with histogram binning.
12. **MARS (Multivariate Adaptive Regression Splines)**: Piecewise linear spline bases via `SplineTransformer` + regularized regression.
13. **Generalized Additive Model (LinearGAM)**: Non-linear univariate penalized B-spline bases via `pygam`.
14. **RBF Network (Radial Basis Function)**: $K$-Means clustering-learned centers $\boldsymbol{\mu}_j$ with Gaussian radial kernels $\phi_j(\mathbf{x}) = \exp(-\gamma \|\mathbf{x} - \boldsymbol{\mu}_j\|^2)$.
15. **Multi-Layer Perceptron (MLP)**: Learned hidden-layer representations with non-linear activation functions.

---

## 🔬 Benchmark Leaderboard (Holdout Test Set)

Evaluated on the final 16 weeks of holdout data (July 13, 2012 to October 26, 2012; 720 store-week observations). Ranked by competition **Weighted Mean Absolute Error (WMAE)**, weighting holiday weeks $5\times$:

| Rank | Model | Model Family | WMAE ($) | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ | MASE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | **Adaptive (Bagging)** | **$40,872.59** | **$39,928.99** | **$64,169.21** | **3.85%** | **0.9850** | **0.7733** |
| **2** | **XGBoost** | **Adaptive (Boosting)** | **$42,455.82** | **$42,167.97** | **$66,945.89** | **4.11%** | **0.9837** | **0.8167** |
| **3** | **Gradient Boosting** | **Adaptive (Boosting)** | **$42,674.07** | **$41,970.39** | **$66,167.76** | **4.09%** | **0.9841** | **0.8128** |
| **4** | **LightGBM** | **Adaptive (Boosting)** | **$45,129.10** | **$43,998.95** | **$67,850.91** | **4.21%** | **0.9833** | **0.8521** |
| **5** | **Ridge Regression** | **Baseline (Linear)** | **$45,240.02** | **$44,414.23** | **$60,799.83** | **4.73%** | **0.9866** | **0.8602** |
| **6** | **Linear Regression** | **Baseline (Linear)** | **$47,558.42** | **$46,435.07** | **$64,783.64** | **5.08%** | **0.9848** | **0.8993** |
| **7** | **Decision Tree** | **Adaptive (Partition)** | **$47,968.88** | **$48,051.60** | **$74,225.00** | **4.68%** | **0.9800** | **0.9306** |
| **8** | **MARS (Splines)** | **Adaptive (Splines)** | **$48,576.21** | **$47,989.49** | **$67,196.53** | **5.27%** | **0.9836** | **0.9294** |
| **9** | **Polynomial Reg (deg 2)**| **Fixed Basis** | **$51,644.33** | **$49,050.12** | **$68,341.98** | **5.01%** | **0.9830** | **0.9499** |
| **10**| **Seasonal Naive** | **Baseline** | **$54,154.80** | **$51,634.91** | **$82,827.98** | **5.35%** | **0.9751** | **1.0000** |
| **11**| **Naive (Lag 1)** | **Baseline** | **$54,810.76** | **$51,644.11** | **$78,653.54** | **5.06%** | **0.9775** | **1.0002** |
| **12**| **Fourier Features** | **Fixed Basis** | **$66,121.90** | **$58,730.06** | **$82,405.05** | **6.23%** | **0.9753** | **1.1374** |
| **13**| **LinearGAM** | **Adaptive (Splines)** | **$225,015.03**| **$218,861.73**| **$278,455.50**| **24.96%**| **0.7184**| **4.2386**|
| **14**| **RBF Network** | **Adaptive (Kernel)** | **$244,921.72**| **$187,221.57**| **$251,876.95**| **30.71%**| **0.7696**| **3.6259**|
| **15**| **MLP Neural Net** | **Adaptive (Sigmoid)** | **$325,252.89**| **$306,543.79**| **$352,195.76**| **41.07%**| **0.5495**| **5.9368**|

---

## 📈 Key Findings & Insights

1. **Tree Ensembles Excel on Tabular Sales Panels**: Random Forest, XGBoost, and Gradient Boosting achieve top accuracy (MAPE $< 4.15\%$), comfortably beating both naive baselines and smooth parametric networks.
2. **Diebold-Mariano Test**: HLN-adjusted DM test comparing Random Forest vs. Ridge Regression yielded $t = -1.0657$ ($p = 0.2865$). Random Forest achieved a **$4,367.43 lower WMAE** (a 9.65% reduction in competition-weighted error).
3. **Leakage-Free Feature Engineering**:
   - Rolling means and standard deviations computed strictly using groupwise `shift(1).rolling()` inside each store.
   - Forward-looking calendar countdown to the next annual occurrence of Thanksgiving and Christmas.
   - Distinct indicators for `markdown_available` (tracking availability) vs. `markdown_active` (active discount).
   - Preprocessing scalers and encoders fitted strictly inside each training fold using `TimeSeriesFoldTransformer`.
4. **12-Week Multi-Step Recursive Forecast**: Iterative forward simulation captures the exponential demand ramp into **Black Friday ($61.53M)** and **Christmas ($70.53M)**, matching historical seasonal dynamics.

---

## 📂 Repository Structure

```
pbl-review/
├── notebooks/
│   └── sales_forecasting_adaptive_basis_models.ipynb  # Master Jupyter/Colab notebook (24 executable cells)
├── src/
│   ├── __init__.py
│   ├── features.py             # Feature engineering, holiday countdowns, store-isolated rolling stats
│   ├── models.py               # 15 model estimator definitions & wrappers
│   ├── evaluate.py             # Evaluation metrics (WMAE, MASE, MAPE) and Diebold-Mariano test
│   ├── run_experiments.py      # Automated experimental pipeline & artifact generator
│   ├── test_pipeline.py        # Automated test suite (6 passing unit tests)
│   ├── eda_analysis.py         # Exploratory data analysis scripts
│   └── inspect_data.py         # Dataset inspection utilities
├── report/
│   └── sales_forecasting_adaptive_basis_models_report.md # Academic research report & 12-slide pitch deck
├── results/
│   ├── figures/                # 19 diagnostic plots and visualization charts (.png)
│   └── tables/                 # 10 benchmark tables and forecasting outputs (.csv)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Zero-leakage git ignore rules
└── README.md                   # Project overview & documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/aadithya200718/pbl-review.git
cd pbl-review

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from Kaggle [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) and place the CSV files in `data/`:
```
data/
├── train.csv
├── features.csv
└── stores.csv
```

### 3. Run Unit Tests
Verify data leakage prevention, store isolation, and feature integrity:
```bash
python -m src.test_pipeline
```

### 4. Run Full Experiments
Train all 15 models, tune hyperparameters, generate diagnostic plots and metrics:
```bash
python -m src.run_experiments
```

### 5. Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/sales_forecasting_adaptive_basis_models.ipynb
```

---

## 📜 Reproducibility & Licensing

- **Random Seed**: Fixed to `42` across all models, cross-validation folds, and random states.
- **Dataset**: Kaggle Walmart Recruiting - Store Sales Forecasting (Public Competition Dataset).
- **License**: MIT License.
