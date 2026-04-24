# Titanic ML Pipeline — Data Leakage Audit

A structured walkthrough of finding and fixing a subtle but consequential data leakage bug in a Titanic survival prediction pipeline.

---

## The Problem

The original pipeline merges the training and test sets before computing imputation statistics:

```python
full_df = pd.concat([train_df, test_df])          # train + test combined
full_df['Age'] = full_df.groupby(...).transform(  # median includes test rows
    lambda x: x.fillna(x.median())
)
```

This means the **median age, median fare, and mode of embarkation port** are all computed using test-set rows — and those values then flow back into filling missing entries in the training set.

That is data leakage: the model's training process is implicitly informed by data it should never have seen.

---

## Why It Matters

Leakage produces a model that looks accurate in validation but underperforms in production. The imputed fill values are slightly "too good" because they were computed with knowledge of the test distribution. For imputation specifically this effect is small — but the habit of combining sets before any transformation is the root cause of far more serious leaks (e.g., scaling, target encoding) in real pipelines.

---

## Repo Structure

| Branch | Description |
|--------|-------------|
| `main` (initial commit) | Original pipeline with the leakage present |
| `feat/leakage-audit` | Audit: documents exactly where and why the leakage occurs — no code changes |
| `fix/data-leakage` | Fix: rewritten preprocessing that uses only training data for all statistics |

---

## The Fix (Preview)

1. Apply feature engineering (title extraction, family size) to train and test **separately** — these are row-level operations, so no statistics cross the boundary.
2. Compute imputation values (`mode`, `median`) from the **training set only**.
3. Store those values and use them to fill missing entries in the test set.

The `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` in the later cell already follow this pattern correctly — the fix brings the earlier imputation steps in line.

---

## Dataset

Standard [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) dataset — `train.csv` (891 rows) and `test.csv` (418 rows).

## How to Run

```bash
pip install pandas numpy scikit-learn
jupyter notebook "Pipeline Demo.ipynb"
```
