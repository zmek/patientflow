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

patientflow, a Python package, converts patient-level predictions into output that is useful for bed managers in hospitals. If you have a predictive model of some outcome for a patient, like admission or discharge from hospital, you can use patientflow to create bed count distributions for a cohort of patients. The package was developed for University College London Hospitals (UCLH) NHS Trust to predict the number of emergency admissions within the next eight hours. The methods generalise to any problem where it is useful to convert patient-level predictions into outcomes for a whole cohort of patients at a point in time. The repository includes a synthetic dataset and a series of notebooks demonstrating the use of the package.

## Background

I'm [Zella King](https://github.com/zmek/), a health data scientist in the Clinical Operational Research Unit (CORU) at University College London. Since 2020, I have worked with University College London Hospital (UCLH) on practical tools to improve patient flow through the hospital.

Hospital bed managers constantly monitor whether they have sufficient beds to meet demand. At specific points during the day they count numbers of inpatients likely to leave, and numbers of new admissions. Their projections about short-term changes are vital because if they anticipate a shortage of bedsd, bed managers must take swift action to mitigate the situation. 

With a team from UCLH, I developed a predictive tool that is now in daily use by bed managers at the hospital. 

The tool we built for UCLH takes a 'snapshot' of patients in the hospital at a point in time, and using data from the hospital's electronic record system, predicts the number of emergency admissions in the next 8 or 12 hours. We are working on predicting discharges in the same way. 

The key principle is that we take data on hospital visits that are unfinished, and predict whether some outcome (admission from A&E, discharge from hospital, or transfer to another clinical specialty) will happen to each of those patients in a window of time. What the outcome is doesn't really matter; the same methods can be used. 

But the true utility of our approach - and the thing that makes it very generalisable - is that we build up from the patient-level predictions into a predictions for a whole cohort of patients at a point in time. That step is what creates useful information for bed managers. They are less interested in whether any individual will need a bed and more interested in the overall number of beds needed, and in which parts of the hospital. They trade in cohort-level data - numbers of beds needed for patients in A&E, number of transfers out of the acute medical unit to other wards, number of patients leaving a certain ward. And they are always only looking a few hours ahead. 

The methods that we developed for UCLH can be used in any hospitals setting where point-in-time predictions about cohorts of patients are useful. We are sharing these methods because we want to make it easier for researchers and analysts in healthcare to create information products that are useful for site and operations managers in hospitals. 

We provide a Python package to make this convenient. The repository includes a set of notebooks with code written in Python and commentary on how to use the package.

We also show a fully worked example of how to predict emergency demand for beds, and demonstrate how we tailored the approach, using the package, to the specific demands of bed managers at UCLH. 

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
