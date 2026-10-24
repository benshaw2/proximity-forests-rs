import pickle
import numpy as np
import pytest
from proximity_forests import ProximityForestClassifier, ProximityForestRegressor, ProximityIsolationForest

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_equal_length_multivariate_public_api(dtype):
    rng=np.random.default_rng(9)
    X=np.ascontiguousarray(rng.normal(size=(36, 2, 12)),dtype=dtype)
    yc=(X[:,0].mean(axis=1)>0).astype(int)
    yr=np.column_stack([X[:,0].mean(axis=1),X[:,1].mean(axis=1)])
    clf=ProximityForestClassifier(n_estimators=5,bootstrap=False,random_state=3,distances=["dtw_i"]).fit(X,yc)
    assert clf.input_layout_=="multivariate" and clf.n_channels_in_==2 and clf.n_timepoints_in_==12
    assert clf.predict(X).shape==(36,) and clf.predict_proba(X).shape==(36,2) and clf.apply(X).shape==(36,5)
    reg=ProximityForestRegressor(n_estimators=5,n_exemplars=2,bootstrap=False,random_state=3,distances=["dtw_d"]).fit(X,yr)
    assert reg.predict(X).shape==(36,2) and reg.apply(X).shape==(36,5)
    iso=ProximityIsolationForest(n_estimators=5,n_exemplars=2,bootstrap=False,random_state=3,distances=["dtw_i"]).fit(X)
    assert iso.anomaly_score(X).shape==(36,) and iso.apply(X).shape==(36,5)
    restored=pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X),clf.predict(X))
