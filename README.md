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

Welcome to the PatientFlow repo, which is designed to support hospital bed management through predictive modelling. The repository shows methods for forecasting short-term bed capacity, a crucial aspect of hospital operations that impacts patient care and resource allocation.

Please note that you are looking at this repo prior to its first release. It is incomplete.

## Objectives

1. Develop code that was originally written for University College London Hospital (UCLH) into a reusable resource following the principles of [Reproducible Analytical Pipelines](https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/)
2. Share the resource with analysts, bed managers and other interested parties in the NHS and other hospital systems
3. Provide training materials to inform and educate anyone who wishes to adopt a similar approach

## Main Features of our modelling approach

- **User led:** This work is the result of close collaboration with operations directors and bed managers in the Coordination Centre, University College London Hospital (UCLH), over four years. What is modelled directly reflects how they work and what is most useful to them.
- **Focused on short-term predictions:** We demonstrate the creation and evaluation of predictive models. The output from these models is a prediction of how many beds with be needed by patients within a short time horizon of (say) 8 hours. (Later we plan to add modules that also predict supply and net bed position over the same period.)
- **Assumes real-time data is available:** Our focus is on how hospitals can make use of real-time data to make informed decisions on the ground. All the modelling here assumes that a hospital has some capacity to run models using real-time (or near to real-time) data in its electronic health record, even if this data is minimal.

## Main Features of this repository

- **Reproducible** - We follow the principles of Reproducible Analytical Pipelines, with the aim that the code can be easily adopted in other settings
- **Accessible** - All the elements are based on simple techniques and methods in Health Data Science and Operational Research. The narrative in the notebooks is intended to be accessible to someone without any knowledge of programming; it should still be possible to follow the approach. We intend that anyone with some knowledge of Python could understand and adapt the code for their use.
- **Practical:** A synthetic dataset, derived from real patient data, is included within this repo in the `data-synthetic` folder. This can be used to step through the modelling process if you want to run the notebooks yourself. So even if your hospital is not set up to do real-time prediction yet, you can still follow the same steps we took. (Note that, if you use the synthetic dataset, the integrity of relationships between variables is not maintained and you will obtain articifically inflated model performance.) UCLH have agreed we can release an anomymised version of real patient data, but not within the repo. To gain access to this, please contact Dr Zella King, contact details below.
- **Interactive:** The repository includes an accompanying set of notebooks with code written on Python, with commentary. If you clone the repo into your own workspace and have an environment within which to run Jupyter notebooks, you will be able to interact with the code and see it running.

## Getting started

- Exploration: Start with the [notebooks README](notebooks/README.md) to get an outline of the notebooks, and read the [patientflow README](src/patientflow/README.md) to understand our intentions for the Python package
- Installation: Follow the instructions below to set up the environment and install necessary dependencies in your own environment
- Configuration: Repurpose config.yaml to configure the package to your own data and user requirements

## About

This project was inspired by the [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by Tom Monks, and is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

### Project Team

Dr Zella King, Clinical Operational Research Unit (CORU), UCL ([zella.king@ucl.ac.uk](mailto:zella.king@ucl.ac.uk))
Jon Gillham, Institute of Health Informatics, UCL
Professor Sonya Crowe, CORU
Professor Martin Utley, CORU

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

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
python -m patientflow.train.emergency_demand --data_folder_name=data-synthetic --uclh=False
```

There are two arguments

- data_folder_name - specifies where to find the data. This should be in a folder named data-xxx directly below the root of the repository
- uclh - tells the package whether the data is the original UCLH data (in which case certain additional features available, including the patient's age in years) or not

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release

## Acknowledgements

This work was funded by a grant from the UCL Impact Funding. We are grateful to the Information Governance team and the Caldicott Guardian at UCLH for agreeing that we can release real patient data.
