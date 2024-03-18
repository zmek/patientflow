# [WORK IN PROGRESS] PatientFlow: Code and training materials for predicting short-term hospital bed capacity using real-time data

Welcome to the PatientFlow repo, which is designed to support hospital bed management through predictive modelling. The repository aims to show methods for forecasting short-term bed capacity, a crucial aspect of hospital operations that impacts patient care and resource allocation.

Please note that you are looking at this repo prior to its first release. It is incomplete. 

## Objectives
1. Develop code that was originally written for University College London Hospital into a reusable resource following the principles of Reproducible Analytical Pipelines
2. Share the resource with analysts, bed managers and other interested parties in the NHS and other hospital systems
3. Provide training materials to inform and educate anyone who wishes to adopt a similar approach

## Main Features of our modelling approach:

- **User led:** This work is the result of close collaboration with operations directors and bed managers in the Coordination Centre, University College London Hospital, over four years. What is modelled directly reflects how they work and what is most useful to them.
- **Focused on short-term predictions:** We demonstrate the creation and evaluation of predictive models. The output from these models is a prediction of how many beds with be needed by patients within a short time horizon of (say) 8 hours. (Later we plan to add modules that also predict supply and net bed position over the same period)
- **Assumes real-time data is available:** Our focus is on how hospitals can make use of real-time data to make informed decisions on the ground. All the modelling here is designed assuming that a hospital has some capacity to run models using real-time (or near to real-time) data in its electronic health record

## Main Features of this repository:

- **Reproducible** - We follow the principles of Reproducible Analytical Pipelines, with the aim that the code can be easily adopted in other settings
- **Accessible** - All the elements are based on simple techniques and methods in Health Data Science and Operational Research. Our intention is that anyone with some knowledge of Python could understand and adapt the code for their use
- **Modular:** The repository is structured into submodules, each intended to predict specific aspects of bed capacity (supply of empty beds, demand for beds and net position in 8 hours' time).
- **Interactive:** The repository includes an accompanying set of notebooks with code written on Python, and notebooks that will be runable on Colab and BinderHub. 
- **Practical:** We will include a dataset, derived from the work we did at University College London Hospital, which can be used to step through the modelling process. This means that, even if your hospital is not set up to do real-time prediction yet, you can still learn from our approach 

## Repository Structure:

- `patientflow`: This will be a Python package contains all the necessary modules and submodules, including predictive models for emergency demand. Later, we will develop modules for predicting discharge, and net bed position
- `notebooks`: This folder contains the notebooks with training materials to support the use of the package. The first two have been written.  
- `LICENSE`
- `README.md` (You are here)
- `environment.yml`
- `requirements.txt`
- `setup.py`

## Getting Started:

1. **[Coming later] Installation:** Follow the instructions in `setup.py` to set up the environment and install necessary dependencies
2. **[Coming later] Configuration:** Utilise `environment.yml` and `requirements.txt` to configure your environment to run these models
3. **Exploration:** Start with the [`notebooks` folder ](../notebooks) to get an outline of the training materials we intend to provide for modelling of emergency demand for beds, and read the  [`patientflow` README ](../patientflow/README.md) to understand our intentions for the package

## Contributing:

One we have released this as a package, we will welcome contributions from the community. At that point, we will add guidelines on how to contribute effectively.

## Acknowlegement:

The work here has been done by researchers at University College London (UCL). UCL provided some financial support to develop the materials shared here. None of this work would be possible without the support and commitment of many colleagues from University College London Hospital.

This repo has been forked from the excellent [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by Tom Monks. 
