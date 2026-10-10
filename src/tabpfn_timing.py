"""How long does TabPFN take at our real data sizes on this machine?"""
import os
import time

import numpy as np
import torch
from tabpfn import TabPFNRegressor

print("CPU cores:", os.cpu_count(), "| CUDA GPU available:", torch.cuda.is_available())

rng = np.random.default_rng(0)
n_features = 10
X_all = rng.normal(size=(5000, n_features))
y_all = 2 * X_all[:, 0] + rng.normal(size=5000)
X_test = X_all[-300:]

for n_train in (500, 2000, 3942):
    model = TabPFNRegressor()
    t0 = time.time()
    model.fit(X_all[:n_train], y_all[:n_train])
    t_fit = time.time() - t0
    t0 = time.time()
    model.predict(X_test)
    t_pred = time.time() - t0
    print(f"train rows {n_train:5d} | fit {t_fit:7.1f}s | predict 300 rows {t_pred:7.1f}s", flush=True)