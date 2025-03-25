# PatientFlow: Predicting demand for hospital beds using real-time data

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/zmek/patientflow/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/zmek/patientflow/actions/workflows/tests.yml
[linting-badge]:            https://github.com/zmek/patientflow/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/zmek/patientflow/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/zmek/patientflow/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/zmek/patientflow/actions/workflows/docs.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/patientflow
[conda-link]:               https://github.com/conda-forge/patientflow-feedstock
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7389--1527-green.svg)](https://orcid.org/0000-0001-7389-1527)

<!-- [pypi-link]:                https://pypi.org/project/patientflow/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/patientflow
[pypi-version]:             https://img.shields.io/pypi/v/patientflow -->
<!-- prettier-ignore-end -->

## Summary

patientflow, a Python package, converts patient-level predictions into output that is useful for bed managers in hospitals.

We developed this code originally for University College London Hospitals (UCLH) NHS Trust to predict the number of emergency admissions within the next eight hours. The methods generalise to other aspects of patient flow in hospitals, including predictions of discharge numbers, within a group of patients. It can be applied to any problem where it is useful to convert patient-level predictions into outcomes for a whole cohort of patients at a point in time.

If you have a predictive model of some outcome for a patient, like admission or discharge from hospital, you can use patientflow to create bed count distributions for a cohort of patients. We show how to prepare your data and train models for these kinds of problems. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.

## What patientflow is for:

- Managing patient flow in hospitals: The package can be used to predict numbers of emergency admissions, discharges or transfers between units
- Short-term operational planning: The predictions produced by this package are designed for bed managers who need to make decisions within an 4-16 hour timeframe.
- Working with real-time data: The design assumes that data from an electronic health record (EHR) is available in real-time, or near to real-time
- Point-in-time analysis: The packages works by taking "snapshots" of groups of patients at a particular moment, and making projections from those specific moments.

## What patientflow is NOT for:

- Long-term capacity planning: The package focuses on immediate operational needs (hours ahead), not strategic planning over weeks or months.
- Making decisions about individual patients: The package is not designed for clinical decision-making about specific patients. It relies on data entered into the EHR by clinical staff looking after patients, but cannot and should not be use to influence their decision-making
- General hospital analytics: It is specifically focused on short-term bed management, not broader hospital analytics like long-term demand and capacity planning.
- Finished/historical patient analysis: While historical data might train underlying models, the package itself focuses on patients currently in the hospital or soon to arrive
- Replacing human judgment: It augments the information available to bed managers, but isn't meant to automate bed management decisions completely.

## This package will help you if you want to:

- Convert individual patient predictions to cohort-level insights: Its core purpose is the creation of aggregate bed count distributions, because bed numbers are the currencly used by bed managers.
- Make predictions for unfinished patient visits: It is designed for making predictions when outcome at the end of the visit are as yet unknown.
- Develop your own predictive models of emergency demand: The package includes a fully worked example of how to convert data from A&E visits into the right structure, and use that data to train models that predict numbers of emergency beds.

## This package will not help you if:

- You work with time series data: patientflow works with snapshots of a hospital visit summarising what is in the patient record up to that point in time
- Your focus is on predicting clinical outcomes: the approach is designed

## Mathematical assumptions underlying the conversion from individual to cohort predictions:

- Independence of patient outcomes: The package assumes that individual patient outcomes are conditionally independent given the features used in prediction.
- Symbolic probability generation: The conversion uses symbolic mathematics (via SymPy) to construct a probability generating function that represents the exact distribution of possible cohort outcomes.
- Bernoulli outcome model: Each patient outcome is modeled as a Bernoulli trial with its own probability, and the package computes the exact probability distribution for the sum of these independent trials.
- Coefficient extraction approach: The method works by expanding a symbolic expression and extracting coefficients corresponding to each possible cohort outcome count.
- Optional weighted aggregation: When converting individual probabilities to cohort-level predictions, the package allows for weighted importance of individual predictions, modifying the contribution of each patient to the overall distribution in specific contexts (eg admissions to different specialties).
- Discrete outcome space: The package assumes outcomes can be represented as discrete counts (e.g., number of admissions) rather than continuous values.

## Getting started

- Exploration: Start with the [notebooks README](notebooks/README.md) to get an outline of what is included in the notebooks, and read the [patientflow README](src/patientflow/README.md) for an overview of the Python package
- Installation: Follow the instructions below to set up the environment and install necessary dependencies in your own environment
- Configuration: Repurpose config.yaml to configure the package to your own data and user requirements

### Prerequisites

`patientflow` requires Python 3.10.

### Installation

patientflow is not yet available on PyPI. To install the latest development version, clone it first (so that you have access to the synthetic data and the notebooks) and then install it.

```sh
git clone https://github.com/zmek/patientflow.git
cd patientflow
pip install -e ".[test]" #this will install the code in test mode

```

Navigate to the patientflow folder and run tests to confirm that the installation worked correctly. This command will only work from the root repository. (To date, this has only been tested on Linux and Mac OS machines. If you are running Windows, there may be errors we don't know about.)

```sh
pytest
```

If you get errors running the pytest command, there may be other installations needed on your local machine. (We have found copying the error messages into ChatGPT or Claude very helpful for diagnosing and troubleshooting these errors.)

### Training models with data provided

The data provided (which is synthetic) can be used to demonstrate training the models. To run training you have two options

- step through the notebooks (for this to work you'll either need copy the two csv files from `data-synthetic`into your `data-public` folder or request access on [Zenodo](https://zenodo.org/records/14866057) to real patient data
- run a Python script using following commands (by default this will run with the synthetic data in its current location; you can change the `data_folder_name` parameter if you have the real data in `data-public`)

```sh
cd src
python -m patientflow.train.emergency_demand --data_folder_name=data-synthetic
```

The data_folder_name specifies the name of the folder containing data. The function expects this folder to be directly below the root of the repository

## Roadmap

- [x] Initial Research
- [x] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release

## About

This project was inspired by the [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by Tom Monks, and is based on a template developed by the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

### Project Team

Dr Zella King, Clinical Operational Research Unit (CORU), University College London ([zella.king@ucl.ac.uk](mailto:zella.king@ucl.ac.uk))
Jon Gillham, Institute of Health Informatics, UCL
Professor Sonya Crowe, CORU
Professor Martin Utley, CORU

## Acknowledgements

This work was funded by a grant from the UCL Impact Funding. We are grateful to the Information Governance team and the Caldicott Guardian at UCLH for agreeing that we can release real patient data.
