# PatientFlow: A forthcoming Python package

Our intention is to release this repository as a Python package that can be installed using common methods like `pip install`

The package will support predictions of bed demand and discharges by providing functions that

- predict patient-level probabilities of admission and discharge, by specialty
- create probability distributions predicting number of beds needed for or vacated by those patients, at different levels of aggregation
- return a net bed position by combining predictions of demand and supply of beds
- evaluate and provide visualisation of the performance of these predictions

The package is intended to serve as a wrapper of the functions typically used for such purposes in the `sklearn` and `scipy` python packages, with additional context to support their application and evaluation in bed management in healthcare

## Modules Overview (in order of their use in a typical modelling workflow)

- `load`: A module for loading configuration files, saved data and trained models
- `prepare`: A module for preparing saved data prior to input into model training
- `train`: A module and submodules for training predictive models
- `predictors`: A module and submodules containing custom predictors developed for the `patientflow` package
- `predict`: A module using trained models for predicting various aspects of bed demand and discharges
- `aggregate`: A module that turns patient-level probabilities into aggregate distributions of bed numbers
- `viz`: A module containing convenient plotting functions to examine the outputs from the above functions

Other modules may follow in future

## Deployment

This package is designed for use in hospital data projects analysing patient flow and bed capacity in short time horizons. The modules can be customised to align with specific hospital requirements
