import sys
import pandas, numpy, sklearn, scipy, xgboost, catboost, shap, optuna, plotly, matplotlib, streamlit, pytest
import tabpfn

print("Python:", sys.version)
for lib in [pandas, numpy, sklearn, scipy, xgboost, catboost, shap, optuna, plotly, matplotlib, streamlit, pytest]:
    print(f"{lib.__name__}: {lib.__version__}")
print("tabpfn imported OK")