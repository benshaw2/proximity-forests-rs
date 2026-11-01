# Out-of-bag results

Classification and regression can compute out-of-bag results during fitting.
Enable them with:

```python
model = ProximityForestClassifier(
    sampling="bootstrap",
    oob_score=True,
    random_state=0,
)
model.fit(X, y)
```

A training observation is out of bag for a tree when it was not selected for
that tree's sample. OOB results aggregate only those trees.

## Classification attributes

After fitting with `oob_score=True`:

- `oob_decision_function_` contains class probabilities;
- `oob_prediction_` contains decoded class labels;
- `oob_vote_counts_` contains the number of contributing trees per observation;
- `oob_score_` is accuracy over observations that received at least one vote.

For multi-output classification, probability matrices are returned as one array
per output.

## Regression attributes

After fitting with `oob_score=True`:

- `oob_prediction_` contains continuous predictions;
- `oob_vote_counts_` contains the number of contributing trees per observation;
- `oob_score_` is the coefficient of determination over observations that
  received at least one vote.

## Missing OOB coverage

An observation can receive zero votes, especially with a small forest. Its OOB
prediction is `NaN`, and fitting emits a runtime warning describing the number
of uncovered observations. Increase `n_estimators` for more complete coverage.

OOB scoring requires a sampling policy capable of leaving observations out of a
tree sample. `sampling="full"` does not provide OOB observations.
