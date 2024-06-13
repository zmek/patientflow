# PatientFlow: Code and training materials for predicting short-term hospital bed capacity using real-time data

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
[pypi-link]:                https://pypi.org/project/patientflow/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/patientflow
[pypi-version]:             https://img.shields.io/pypi/v/patientflow
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

Welcome to the PatientFlow repo, which is designed to support hospital bed management through predictive modelling. The repository shows methods for forecasting short-term bed capacity, a crucial aspect of hospital operations that impacts patient care and resource allocation.

Please note that you are looking at this repo prior to its first release. It is incomplete.

## Objectives

1. Develop code that was originally written for University College London Hospital into a reusable resource following the principles of [Reproducible Analytical Pipelines](https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/)
2. Share the resource with analysts, bed managers and other interested parties in the NHS and other hospital systems
3. Provide training materials to inform and educate anyone who wishes to adopt a similar approach

## Main Features of our modelling approach

- **User led:** This work is the result of close collaboration with operations directors and bed managers in the Coordination Centre, University College London Hospital (UCLH), over four years. What is modelled directly reflects how they work and what is most useful to them.
- **Focused on short-term predictions:** We demonstrate the creation and evaluation of predictive models. The output from these models is a prediction of how many beds with be needed by patients within a short time horizon of (say) 8 hours. (Later we plan to add modules that also predict supply and net bed position over the same period.)
- **Assumes real-time data is available:** Our focus is on how hospitals can make use of real-time data to make informed decisions on the ground. All the modelling here assumes that a hospital has some capacity to run models using real-time (or near to real-time) data in its electronic health record, even if this data is minimal (see next point).
- **Demonstrates prediction with minimal data:** Recognising that some hospitals are not set up for real-time data modelling, we also demonstrate how short-term demand forecasting could be done using only the datapoints collected for the [Operational Pressures Escalation Levels (OPEL) Framework](https://www.england.nhs.uk/wp-content/uploads/2016/10/PRN00551-OPEL-Framework-2023.24-V2.0.pdf)

## Main Features of this repository

- **Reproducible** - We follow the principles of Reproducible Analytical Pipelines, with the aim that the code can be easily adopted in other settings
- **Accessible** - All the elements are based on simple techniques and methods in Health Data Science and Operational Research. The narrative in the notebooks is intended to be accessible to someone without any knowledge of programming; it should still be possible to follow the approach. We intend that anyone with some knowledge of Python could understand and adapt the code for their use.
- **Modular:** The repository is structured into submodules, each intended to predict specific aspects of bed capacity (supply of empty beds, demand for beds and net position in 8 hours' time).
- **Interactive:** The repository includes an accompanying set of notebooks with code written on Python, and notebooks that will be runable on Colab and BinderHub.
- **Practical:** We hope to include a dataset, derived from the work we did at University College London Hospital, which can be used to step through the modelling process. This means that, even if your hospital is not set up to do real-time prediction yet, you can still follow the same steps we took. We are currently working on a Data Protection Impact Assessment (DPIA) with our colleagues at UCLH.

This project was inspired by the excellent [py-pi template](https://github.com/health-data-science-OR/pypi-template) developed by Tom Monks, and is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About

### Project Team

Zella King ([zella.king@ucl.ac.uk](mailto:zella.king@ucl.ac.uk))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

Start with the [`notebooks` README](./notebooks/README.md) to get an outline of the training materials we intend to provide for modelling of emergency demand for beds, and read the [`patientflow` README ](./src/patientflow/README.md) to understand our intentions for the Python package

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`patientflow` requires Python 3.6.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `patientflow` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/zmek/patientflow.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/zmek/patientflow.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

How to run the application on your local system.

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release

## Acknowledgements

This work was funded by a grant from the UCL Impact Funding.
