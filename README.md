# PatientFlow: Code and explanatory notebooks for predicting short-term hospital bed capacity using real-time data

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

Welcome to the PatientFlow repository, which is designed to support hospital bed management through predictive modelling. I'm [Zella King](https://github.com/zmek/), a health data scientist in the Clinical Operational Research Unit (CORU) at University College London. Since 2020, I have worked with University College London Hospital (UCLH) NHS Trust on practical tools to improve patient flow through the hospital.

My most important contribution is a software application that [Jon Gillham](https://github.com/jongillham) and I developed, which is now in daily use by bed managers at the hospital. That application generates predictions of emergency demand for beds, using real-time data from the hospital's patient record system, and sends the predictions to the bed managers. I created the predictive models that are used in the application. Jon created the software that runs my modelling code five times a day, and sends the predictions by email to the bed managers.

I developed the code I wrote for UCLH into a reusable resource following the principles of [Reproducible Analytical Pipelines](https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/). I did this because I want to:

1. Share the code with researchers and NHS analysts who are working on similar models
2. Make it easier for others to make use of the mathematics involved in making these predictions
3. Inform and educate anyone who wishes to adopt a similar approach

## Main features of my modelling approach

- **Led by what users need:** My work is the result of close collaboration with operations directors and bed managers in the Coordination Centre, University College London Hospital (UCLH), since 2020. What is modelled directly reflects how they work and what is most useful to them.
- **Focused on short-term predictions:** I am expert in predicting demand within a short time horizon eg 8 or 12 hours. Here I show models that predict how many beds will be needed emergency patients. (Later I plan to add modules that also predict elective demand, discharge and transfers between specialties.)
- **Assumes real-time data is available:** Hospital bed managers have to deal with rapidly changing situations. My focus is on the use of real-time data (or near to real-time) to help them make informed decisions. The modelling shown here assumes that a hospital has some capacity to make use of real-time data in its electronic health record, even if this data is minimal.

## Main Features of this repository

- **Reproducible** - I follow the principles of [Reproducible Analytical Pipelines](https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/). The repository can be installed as a Python package, and imported into your own code.
- **Accessible** - All the elements are based on simple techniques and methods in Health Data Science and Operational Research. I intend that anyone with some knowledge of Python could understand and adapt the code for their use.
- **Practical:** - I believe that it is easier to follow the steps I took if you have access to the same data I have. UCLH have released an anonymised version of real patient data, which you can request access to on [Zenodo](https://zenodo.org/records/14866057), or you can use the synthetic dataset, derived from real patient data, in the `data-synthetic` folder. (Note that, if you use the synthetic dataset, the integrity of relationships between variables is not maintained and you will obtain articifically inflated model performance.)
- **Interactive:** The repository includes a set of notebooks with code written on Python, with commentary. If you clone the repo into your own workspace and have an environment for running Jupyter notebooks, you will be able to interact with the code and see it running.

## Getting started

- Exploration: Start with the [notebooks README](notebooks/README.md) to get an outline of the notebooks, and read the [patientflow README](src/patientflow/README.md) to understand my intentions for the Python package
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

- step through the notebooks (for this to work you'll either need copy the two csv files from `data-synthetic`into your `data-public` folder or contact us for real patient data)
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
