# PatientFlow: A forthcoming Python package

Our intention is to release this folder, and its subfolders, as a Python package that can be installed using common methods like `pip install`.  

## Modules Overview:

- `predict`: The central module containing submodules for predicting various aspects of bed capacity
  - `emergencydemand`: Forecasts the number of emergency beds required

Other modules may follow in future.

## Key Objectives:

- **Aggregate-Level prediction:** Focus on predicting the overall number of beds needed
- **Specialty and sex Breakdown:** Provide predictions by specialty and sex, allowing for targeted resource allocation
- **Inclusion of future arrivals:** Factor in patients who are yet to arrive, thus providing a more complete picture of demand

## Utilization:

This package is designed for use in hospital data systems, enabling real-time prediction and analysis. The modules can be customised to align with specific hospital requirements
