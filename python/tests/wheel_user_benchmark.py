from time import perf_counter
import numpy as np
from proximity_forests import ProximityForestClassifier

rng = np.random.default_rng(42)
X = np.ascontiguousarray(rng.normal(size=(300, 64)), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
model = ProximityForestClassifier(n_estimators=25, n_candidates=5, sampling="full", random_state=42, n_jobs=-1, distances=["dtw"])
start = perf_counter(); model.fit(X, y); fit_seconds = perf_counter() - start
start = perf_counter(); prediction = model.predict(X); predict_seconds = perf_counter() - start
assert prediction.shape == (X.shape[0],)
print(f"fit_seconds={fit_seconds:.6f}")
print(f"predict_seconds={predict_seconds:.6f}")
print("NOTE: smoke timings only; not a comparative benchmark")
