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

`patientflow`, a Python package, converts patient-level predictions into output that is useful for bed managers in hospitals.

We originally developed this code for University College London Hospitals (UCLH) to predict the number of emergency admissions they should expect within the next eight hours. Our method used real-time data from their Electronic Health Record (EHR) system. We wrote code to convert patient-level data, extracted from the EHR at a point in time, into predicted numbers of admissions in the following 4 or 8 hours. We also wrote code to help us evaluate the predictions.

We have created the `patientflow` python package to make it convenient for others to adopt our approach. Its purpose is to predict bed demand for groups of hospital patients at a point in time. The package is organised around the following concepts:

- Prediction time: A moment in the day at which predictions are to be made, for example 09:30.
- Patient snapshot: A summary of data from the EHR capturing is known about a single patient at the prediction time. Each patient snapshot has a date and a prediction time associated with it.
- Group snaphot: A set of patients snapshots. Each group snapshot has a date and a prediction time associated with it.
- Prediction window: A period of hours that begins at the prediction time.

The modelling functions in `patientflow` are designed to receive a group snapshot as an input, and to predict something about that group's demand for beds between the prediction moment and the end of the prediction window. For example, that group could be the patients currently in the Emergency Department (ED), and the predictions could be the number of beds needed by those patients in the next 8 hours. The output is a probability distribution over the number of beds needed. The package includes functions to generate predictions at both patient and group level, to visualise predicted probability distributions, and to evaluate them.

This snapshot-based approach to predicting demand generalises to other aspects of patient flow in hospitals, such as predictions of how many patients from a clinical specialty will be discharged. A series of notebooks demonstrates the use of the package. We show how to prepare your data and train models based on a snapshot approach. The repository includes a synthetic dataset, and an anonymised patient dataset, based on real data from UCLH is available on [Zenodo](https://zenodo.org/records/14866057). Both the synthetic and the real dataset have been prepared in a snapshot structure.

## What `patientflow` is for:

- Predicting patient flow in hospitals: The package can be used by researchers or analysts who want to predict numbers of emergency admissions, discharges or transfers between units
- Short-term operational planning: The predictions produced by this package are designed for bed managers who need to make decisions within an 4-16 hour timeframe.
- Working with real-time data: The design assumes that data from an electronic health record (EHR) is available in real-time, or near to real-time
- Point-in-time analysis: The package works by taking snapshots of groups of patients who are in the hospital at a particular moment, and making predictions about whether a non-clinical outcome like admission or discharge will occur with a short time horizon.

## What `patientflow` is NOT for:

- Long-term capacity planning: The package focuses on immediate operational demand (hours ahead), not strategic planning over weeks or months.
- Making decisions about individual patients: The package relies on data entered into the EHR by clinical staff looking after patients, but the patient-level predictions it generates should not be used to influence their decision-making
- General hospital analytics: The package is designed for short-term bed management, not broader hospital analytics like long-term demand and capacity planning.
- Predicting what happens _after_ a hospital visit: While historical data might train underlying models, the package itself focuses on patients currently in the hospital or soon to arrive
- Replacing human judgment: The predictions are meant to augment the information available to bed managers, but not to automate bed management decisions.

## This package will help you if you want to:

- Make predictions for unfinished patient visits: The package is designed for making predictions when outcomes at the end of the visit are as yet unknown, and evaluating those predictions against what actually happened.
- Convert individual patient predictions to group-level insights: As bed numbers are the currency used by bed managers, the package generates bed count distributions; you may find this kind of output will help you interest hospital site and operations managers in your predictions.
- Develop your own predictive models of emergency demand: The repository includes a fully worked example of how to convert historical data from Emergency Department visits into snapshots, and use the snapshots to train models that predict numbers of emergency beds.

## This package will NOT help you if:

- You work with time series data: `patientflow` works with snapshots of a hospital visit summarising what is in the patient record up to that point in time. It would need modification to accept time series data formats.
- You want to predict clinical outcomes: the approach is designed for the management of hospital sites, not the management of patient care.

## Mathematical assumptions underlying the conversion from individual to group predictions:

- Independence of patient requirements: The package assumes that individual patient requirements (eg for admission) are conditionally independent.
- Bernoulli outcome model: Each patient outcome is modeled as a Bernoulli trial with its own probability, and the package computes a probability distribution for the sum of these independent trials.
- Different levels of aggregation: The package can calculate probability distributions for compound scenarios (such as the probability of a patient being admitted, assigned to a specific specialty if admitted, and being admitted within the prediction window) and for patient subgroups (like distributions by age or gender). In all cases, the independence assumption between patients is maintained.

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

If you get errors running the pytest command, there may be other installations needed on your local machine.

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

This idea to create a Python package was inspired by , and

### Project Team

- [Dr Zella King](https://github.com/zmek), Clinical Operational Research Unit (CORU), University College London ([zella.king@ucl.ac.uk](mailto:zella.king@ucl.ac.uk))
- [Jon Gillham](https://github.com/jongillham), Institute of Health Informatics, UCL
- Professor Sonya Crowe, CORU
- Professor Martin Utley, CORU

## Acknowledgements

The [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by [Tom Monks](https://github.com/TomMonks) inspired us to create a Python package. This repository is based on a template developed by the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London. We are grateful to [Lawrence Lai](https://github.com/lawrencelai) for creation of the synthetic dataset. MAPS QR Policy Funding from by University College London contributed to the construction of the repository.

The underlying academic work was funded by grants from

- the Wellcome Institutional Strategic Support Fund (ISSF) UCL and Partner Hospitals: AI in Healthcare Funding Call 2019 (award number BRC717/HI/RW/101440),
- the National Institute for Health Research UCLH Biomedical Research Centre HIGODS Theme (award number BRC824/HG/ZK/110420)
- the National Institute for Health Research (Artificial Intelligence, Digitally adapted, hyper-local realtime bed forecasting to manage flow for NHS wards, AI_AWARD01786) and NHSX
- University College London Hospitals NHS Trust (Zetetic Benefits-Enhancing Data Science)
