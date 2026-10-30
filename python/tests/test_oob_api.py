import pickle
import warnings
import numpy as np
import pytest
from proximity_forests import ProximityForestClassifier, ProximityForestRegressor


def data():
    rng=np.random.default_rng(660)
    X=np.ascontiguousarray(rng.normal(size=(96,8)))
    yc=np.column_stack(((X[:,0]>0).astype(int),(X[:,1]>0).astype(int)))
    yr=np.column_stack((X[:,0]+X[:,1],X[:,2]-X[:,3]))
    return X,yc,yr

@pytest.mark.parametrize("kind",["classifier","regressor"])
def test_oob_attributes_parallel_equivalence_and_pickle(kind):
    X,yc,yr=data(); common=dict(n_estimators=17,n_candidates=3,sampling="bootstrap",random_state=66,oob_score=True)
    cls=ProximityForestClassifier if kind=="classifier" else ProximityForestRegressor; y=yc if kind=="classifier" else yr
    one=cls(n_jobs=1,**common).fit(X,y); many=cls(n_jobs=4,**common).fit(X,y)
    np.testing.assert_array_equal(one.oob_vote_counts_,many.oob_vote_counts_)
    np.testing.assert_equal(one.oob_prediction_,many.oob_prediction_)
    assert one.oob_score_==many.oob_score_ or (np.isnan(one.oob_score_) and np.isnan(many.oob_score_))
    if kind=="classifier":
        for a,b in zip(one.oob_decision_function_,many.oob_decision_function_): np.testing.assert_array_equal(a,b)
    restored=pickle.loads(pickle.dumps(many))
    np.testing.assert_equal(restored.oob_prediction_,many.oob_prediction_)
    np.testing.assert_array_equal(restored.oob_vote_counts_,many.oob_vote_counts_)

def test_disabled_oob_has_no_attributes():
    X,y,_=data(); model=ProximityForestClassifier(n_estimators=3,sampling="full",oob_score=False).fit(X,y[:,0])
    assert not hasattr(model,"oob_score_")

def test_zero_votes_warn_and_are_nan():
    X,y,_=data()
    with pytest.warns(RuntimeWarning, match=r"96 of 96 training samples received no OOB votes"):
        model=ProximityForestClassifier(n_estimators=3,sampling="full",oob_score=True).fit(X,y[:,0])
    assert np.isnan(model.oob_score_); assert np.isnan(model.oob_decision_function_).all(); assert np.all(model.oob_vote_counts_==0)
