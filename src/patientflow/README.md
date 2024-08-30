# PatientFlow: A forthcoming Python package

Our intention is to release this folder, and its subfolders, as a Python package that can be installed using common methods like `pip install`

The package will support predictions of bed demand and discharges by providing functions that

- take as input patient-level probabilities of admission and discharge
- return as output probability distributions predicting number of beds needed for or vacated by those patients, at different levels of aggregation (eg by specialty or sex)
- return a net bed position by combining predictions of demand and supply of beds
- provide visualisation of the performance of these predictions using qq plots

The package is intended to serve as a wrapper of the functions typically used for such purposes in the `scipy` python package, with additional context to support their application and evaluation in bed management in healthcare

## Modules Overview (in order of their use)

- `load`: A module for loading saved data and trained models
- `prepare`: A module for preparing saved data prior to input into model training
- `train`: A module and submodules for training models
- `predictors`: A module and submodules containing customer predictors used in patientflow
- `predict`: A module using trained models for predicting various aspects of bed capacity
- `aggregate`: A module that turns patient-level probabilities into aggregate distributions
- `viz`: A module containing convenient plotting functions to examine the outputs from the above functions

Other modules may follow in future

## Deployment

This package is designed for use in hospital data projects analysing patient flow and bed capacity in short time horizons. The modules can be customised to align with specific hospital requirements
