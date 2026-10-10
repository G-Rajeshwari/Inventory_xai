"""Smoke test: does TabPFN install, download weights and predict on this machine?"""
import inspect
import time
from importlib.metadata import version

import numpy as np
from tabpfn import TabPFNRegressor

print("tabpfn version:", version("tabpfn"))
print("TabPFNRegressor parameters:")
print(inspect.signature(TabPFNRegressor.__init__))

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 8))
y = 2 * X[:, 0] + rng.normal(size=300)

t0 = time.time()
model = TabPFNRegressor()
model.fit(X[:250], y[:250])
pred = model.predict(X[250:])
secs = time.time() - t0

print("\nPredictions (first 5):", np.round(pred[:5], 3))
print("Actual      (first 5):", np.round(y[250:255], 3))
print(f"Correlation prediction vs actual: {np.corrcoef(pred, y[250:])[0, 1]:.3f}")
print(f"Time for fit + predict on 250 rows: {secs:.1f} seconds")