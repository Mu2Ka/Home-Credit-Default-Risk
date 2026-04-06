# Home Credit Default Risk

This project is a machine learning study on the Home Credit Default Risk dataset. The goal is to predict whether a client will have payment difficulties using application data and several related credit history tables.

## What the project includes

- exploratory data analysis in `main.ipynb`
- feature engineering from:
  - `bureau`
  - `bureau_balance`
  - `credit_card_balance`
  - `installments_payments`
- preprocessing of missing values and categorical features
- baseline model comparison

## What has been done

- loaded core tables through reusable Python modules in `src/`
- moved part of feature engineering from the notebook into Python files
- built aggregated features from external credit history tables
- compared several baseline models:
  - Logistic Regression
  - Random Forest
  - LightGBM
  - CatBoost
- evaluated models with:
  - ROC-AUC
  - PR-AUC
  - ROC curves
  - Precision-Recall curves

## Current best result

At the current baseline stage, LightGBM shows the strongest quality among the tested models, with CatBoost performing very close behind.

## Project structure

- `main.ipynb` - main notebook with EDA, preprocessing, training, and evaluation
- `src/data_loader.py` - loading project tables
- `src/features.py` - feature engineering functions
- `src/example_pipeline.py` - small example of using Python modules

## Next steps

- improve feature engineering
- clean up preprocessing
- tune LightGBM and CatBoost
- add a cleaner training pipeline and final submission generation
