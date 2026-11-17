modules = [
    'pandas', 'numpy', 'scipy', 'sklearn', 'statsmodels',
    'xgboost', 'lightgbm', 'matplotlib', 'seaborn', 'shap',
    'pygam', 'pylearn', 'prophet'
]

for mod in modules:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'available')
        print(f"OK   {mod:15s}: {ver}")
    except ImportError as e:
        print(f"MISS {mod:15s}: {e}")
