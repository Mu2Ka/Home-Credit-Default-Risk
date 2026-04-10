# Home Credit Default Risk

Machine learning project for the Kaggle competition
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

The goal is to estimate the probability that a credit applicant will have
payment difficulties. In this dataset, `TARGET = 1` means that the client had
payment difficulties, while `TARGET = 0` means no such difficulties were
recorded.

The project is built around `main.ipynb`, which contains the full modeling
workflow from raw Kaggle CSV files to `submission.csv`.

## Notebook Workflow

The notebook follows these steps:

1. Load the core Home Credit tables.
2. Aggregate external tables to the client level and merge them into train and
   test datasets.
3. Explore the target distribution and selected categorical and numerical
   features.
4. Clean missing values, special values, categorical variables, and infinite
   values.
5. Build additional ratio and history features.
6. Train baseline models.
7. Compare ROC-AUC and PR-AUC / Average Precision.
8. Select the most important LightGBM features.
9. Tune LightGBM with Optuna.
10. Train the final model and create `submission.csv`.

Because the target is highly imbalanced, PR-AUC / Average Precision is treated
as an important metric alongside ROC-AUC.

## Data Sources

The notebook uses the following Kaggle files:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `previous_application.csv`
- `HomeCredit_columns_description.csv`
- `sample_submission.csv`

The additional tables are aggregated by `SK_ID_CURR`, so each final row still
represents one client application.

After merging the external features, the current notebook run produced:

- train shape: `(307511, 236)`
- test shape: `(48744, 235)`
- encoded train/validation split: `(246008, 382)` and `(61503, 382)`

## Feature Engineering

Feature engineering is split between reusable functions in `src/features.py`
and the `FEATURE_ENGINEERING` function inside `main.ipynb`.

The project builds aggregated features from:

- credit bureau history
- monthly bureau balance status
- credit card balances
- installment payment history
- POS/CASH loan history
- previous loan applications

The notebook also creates modeling features such as:

- credit-to-income ratio
- annuity-to-income ratio
- income per family member
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` aggregate statistics
- indicators for previous POS history, previous applications, and approved
  applications
- indicators for credit bureau request activity

Missing values are filled with a mix of means, medians, zeros, `Unknown`, and
special sentinel values depending on the feature group.

## Models

The notebook compares four baseline models:

- Logistic Regression
- Random Forest
- LightGBM
- CatBoost

Categorical features are One-Hot Encoded for most models. CatBoost is trained
separately with native categorical feature handling.

After baseline training, LightGBM feature importance is used to run a feature
count experiment. The notebook tests feature counts from 1 to 382 and then keeps
the top 150 features for tuning and final training.

Selected feature-count results from the current run:

| Number of features | ROC-AUC | PR-AUC |
| ---: | ---: | ---: |
| 50 | 0.77526 | 0.26822 |
| 80 | 0.77991 | 0.27349 |
| 120 | 0.78092 | 0.27610 |
| 150 | 0.78178 | 0.28004 |
| 200 | 0.78282 | 0.27927 |
| 382 | 0.78216 | 0.27973 |

The top 150 features were chosen as a compact feature set with strong
validation quality.

## Current Result

LightGBM is tuned with Optuna using `average_precision` as the optimization
target.

Current Optuna run:

- requested trials: `100`
- completed trials: `68`
- timeout: `4000` seconds
- best Optuna Average Precision: `0.27290`
- validation ROC-AUC after tuning: `0.78669`
- 5-fold CV Average Precision: `0.27586 +/- 0.00627`

Best parameters from the current notebook run:

```text
max_depth: 9
n_estimators: 1381
learning_rate: 0.011802717142163873
num_leaves: 49
min_child_samples: 46
subsample: 0.8206620477389999
subsample_freq: 3
colsample_bytree: 0.5038096663449956
reg_alpha: 6.788492363468327
reg_lambda: 2.4916010300837357
min_split_gain: 0.6980155449785576
```

The final notebook cell trains LightGBM on the full training data with the
selected features and writes predictions to `submission.csv`.

## Repository Structure

```text
.
|-- main.ipynb              # EDA, preprocessing, training, evaluation
|-- requirements.txt        # Python dependencies
|-- src/
|   |-- data_loader.py      # CSV loading helpers
|   |-- features.py         # Feature engineering functions
|   `-- example_pipeline.py # Minimal example pipeline
`-- README.md
```

## Data

The competition CSV files are not stored in Git because they are large. Download
the dataset from Kaggle and place the files in the project root:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
credit_card_balance.csv
installments_payments.csv
POS_CASH_balance.csv
previous_application.csv
HomeCredit_columns_description.csv
sample_submission.csv
```

Generated files such as `submission.csv`, local archives, model artifacts, and
CatBoost training logs are ignored by Git.


