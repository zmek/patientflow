# PatientFlow: A forthcoming Python package

Our intention is to release this folder, and its subfolders, as a Python package that can be installed using common methods like `pip install`

The package will support predictions of bed demand by providing functions that

- take as input patient-level probabilities of admission and discharge
- return as output probability distributions predicting number of beds needed for or vacated by those patients, at different levels of aggregation (eg by specialty or sex)
- return a net bed position by combining predictions of demand and supply of beds
- provide visualisation of the performance of these predictions using qq plots

The package is intended to serve as a wrapper of the functions typically used for such purposes in the `scipy` python package, with additional context to support their application and evaluation in bed management in healthcare

## Modules Overview

- `predict`: The central module containing submodules for predicting various aspects of bed capacity
  - `emergencydemand`: generate predictions of the number of emergency beds required within a short time horizon. See [this notebook](../notebooks/2%20Introducing%20emergency%20demand%20and%20its%20modelling.ipynb) for an introduction to the modelling approach
  - [Later] `emergencysupply`: generate predictions of the number of emergency beds that will become free within a short time horizon due to patients being discharged
  - [Later] `net bed position`: using the above functions, generate predictions of the net bed position (surplus or deficit of beds) within a short time horizon
- `viz`: A module containing convenient plotting functions to examine the outputs from the above functions

Other modules may follow in future

## Deployment

This package is designed for use in hospital data projects analysing patient flow and bed capacity in short time horizons. The modules can be customised to align with specific hospital requirements
