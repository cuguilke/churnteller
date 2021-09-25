# Customer Churn Prediction

This repository contains codes for customer churn prediction

## Table of Contents

1. [Dependencies](#dependencies)
2. [Arguments](#arguments)
3. [Data](#data)
4. [Hyperparameters](#hyperparameters)
5. [Features](#features)
6. [Tests](#tests)
7. [Experiments](#experiments)
8. [Results](#results)

## Dependencies

Install the dependencies using `pip`

```
pip install -r requirements.txt
```

## Arguments

- `model`: XGBoost or SVM
- `features`: RFM (Recency, Frequency, Monetary model) or custom (includes payment, platform, and transmission type frequencies)
- `n_splits`: the number of folds for the final cross-validation results
- `normalize`: to enable feature normalization (needed for SVM trainings)
- `grid_search`: to run grid search to find out optimal hyperparameters using 10 fold CV
- `print_config`: to print out the selected configuration

## Data

1. The time interval between `2016-09-01` and `2017-02-28` is used to form new labels for training.
2. 10-fold cross-validation is used to find and store the best performing hyperparameters for a given model.
3. Final models are trained with customer data from `2015-03-01` to `2016-09-01`.
4. Then, they are tested using full customer data (`2015-03-01` to `2017-02-28`) and the given labels.

## Hyperparameters

[`parameters.json`](parameters.json) file contains the stored hyperparameters for the available models

To search for new hyperparameters modify [`models.py`](models.py) and run:

```
python run.py --model <xgboost,svm> --features <RFM,custom> --grid_search
```

## Features

RFM model contains:
- `Recency`: the time elapsed since last purchase
- `Frequency`: the total number of orders made in a time interval
- `Monetary`: the total amount of money spend in a time interval

Custom model contains (in addition to RFM features):
- `Payment ID Frequency`: the total number of preference per payment type
- `Platform ID Frequency`: the total number of preference per platform type
- `Transmission ID Frequency`: the total number of preference per transmission type

## Tests

To run the tests:

```
python -m unittest discover -s test
```

## Experiments

- RFM data model (Bult & Wansbeek, 1995): SVM is worse than XGBoost for all metrics

|         |  Acc | Precision | Recall | F1-score |
|---------|:----:|:---------:|:------:|:--------:|
| SVM     | 0.71 ± 0.14 |    0.78 ± 0.02   |  0.86 ± 0.25  |   0.79 ± 0.19   |
| XGBoost | 0.84 ± 0.00 |    0.85 ± 0.00   |  0.96 ± 0.00  |   0.90 ± 0.00   |

- Normalized RFM data model: SVM performance suffers after feature normalization

|         |  Acc | Precision | Recall | F1-score |
|---------|:----:|:---------:|:------:|:--------:|
| SVM     | 0.63 ± 0.13 |    0.77 ± 0.03   |  0.73 ± 0.18  |   0.74 ± 0.11   |
| XGBoost | 0.84 ± 0.00 |    0.85 ± 0.00   |  0.96 ± 0.00  |   0.90 ± 0.00   |

- Custom data model: (includes payment, platform, transmission type frequency per customer) SVM cannot learn with these features without normalization 

|         |  Acc | Precision | Recall | F1-score |
|---------|:----:|:---------:|:------:|:--------:|
| SVM     | 0.38 ± 0.08 |    0.77 ± 0.04   |  0.28 ± 0.10 |   0.41 ± 0.10  |
| XGBoost | 0.84 ± 0.00 |    0.85 ± 0.00   |  0.96 ± 0.00 |   0.90 ± 0.00  |

- Normalized custom data model: SVM results are fixed after feature normalization and it became competitive with XGBoost

|         |  Acc | Precision | Recall | F1-score |
|---------|:----:|:---------:|:------:|:--------:|
| SVM     | 0.80 ± 0.00 |    0.82 ± 0.01  |  0.95 ± 0.01 |   0.88 ± 0.00  |
| XGBoost | 0.84 ± 0.00 |    0.85 ± 0.00  |  0.96 ± 0.00 |   0.90 ± 0.00  |

## Results

- Customer churn rate for this dataset is 77%, due to the this imbalance, the test accuracy itself may give a false impression about the predictive performance.
- XGBoost has high recall (important since we want to predict customer churn beforehand to prevent them from leaving) and high precision (meaningful alerts regarding customer likelihood to leave the platform) acroos different data models. 
- XGBoost has quite robust performance across different features/normalization schemes.
- In addition, training XGBoost has significantly lower computational cost than that of SVM. Hence, one can perform extensive grid search for optimal hyperparameters.
- The best solution can be reproduced using this command:

```
python run.py --model xgboost --features custom --normalize
```
