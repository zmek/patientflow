# PatientFlow: A forthcoming Python package

Our intention is to release this folder, and its subfolders, as a Python package that can be installed using common methods like `pip install`.  

## Modules Overview:

- `predict`: The central module containing submodules for predicting various aspects of bed capacity
  - `emergencydemand`: Includes functions used to generate predictions of the number of emergency beds required. See [this notebook](../notebooks/notebooks/2%20Introducing%20emergency%20demand%20and%20its%20modelling.ipynb) for an introduction to the modelling approach

Other modules may follow in future.

## Deployment:

This package is designed for use in hospital data projects analysing patient flow and bed capacity in short time horizons. The modules can be customised to align with specific hospital requirements
