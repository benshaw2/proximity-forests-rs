import time, numpy as np
from proximity_forests import ProximityForestClassifier
rng=np.random.default_rng(661); X=np.ascontiguousarray(rng.normal(size=(512,16))); y=(X[:,0]>0).astype(int)
for enabled in (False,True):
 t=[]
 for seed in range(5):
  start=time.perf_counter(); ProximityForestClassifier(n_estimators=50,n_candidates=5,sampling="bootstrap",random_state=seed,n_jobs=4,oob_score=enabled).fit(X,y); t.append(time.perf_counter()-start)
 print(f"oob_score={enabled}: median={np.median(t):.6f}s runs={t}")
