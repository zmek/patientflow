# pypi-template

Notes and a template repo for PyPi Projects.

T.Monks. Jan 2021.


## Useful posts for learning:
* https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/#what-is-pip

* https://nrempel.com/blog/how-publish-python-package-pypi-comprehensive-guide/

## What is pip?

A package management system for installing python packages from the Python package index (PyPi - pronouced Pie Pie).  Example useage:

`pip install numpy`

or

`pip install numpy==1.18.0`

## Why use PyPi for your own projects?

It may be overkill, but its useful if you want to use a package in multiple projects without including the full source code.  

The obvious use if you want others to be able to easily install and use your software/project in their own work.  I use it for educational software for students at the moment.  Its something they can use after they leave as well.

## Setting up a git repo for a PyPi project.

I've used this structure on a previous project and it other projects I found use similar variations.  

```
pypi_template
├── LICENSE
├── test_package
│   ├── __init__.py
│   ├── test.py
│   ├── data
│   |   ├── test_data.csv
├── README.md
├── environment.yml
├── requirements.txt
└── setup.py
```

### environment.yml

One thing I learnt from attending the Open Source project sessions at SciPy 2019 was that it is useful to include a conda environment for developers.  That is what is in `environment.yml`. One ommision in the template repo is that it is useful to install your package into the dev environment!  See for example:  https://github.com/TomMonks/forecast-tools 

### setup.py

This controls the installation of your package.  I've included a template in the repo.  We need to use the `setuptools` PyPI package to do the installation.  I've included `setuptools` in the dev environment, but you can install it manually:

`pip install setuptools`

Take a look at setup.py for details.

### including data in your package

For data science project you may want to include some example or test data within your package. 

As an example, the package in the template repo includes a subdirectory `test_package/data` containing a single (dummy) CSV.

One thing I learnt **the hard and annoying way** is that data is not included in your python package by default!  

To make sure data is included I took two steps

#### Tell `setup()` that it is expecting data

There are a few lines of code to include in the setup function.

```python
    #Tells setup to look in MANIFEST.in
    include_package_data=True,
    #optional - can be used for file not found in 
    #MANIFEST.in
    package_data={"test_package": ["data/*.csv"]},
```

* RTM: https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html 

#### Include a MANIFEST.in file

```
include *.txt
recursive-include test_package/data *.csv
```

### local installing and uninstalling your package

Now that we have a setup.py and have installed `setuptools` we can use them to install our package locally!  Navigate to the repo on your local machine and run the command below.

`pip install .`

**Exercise**: Test out your install by launching the python interpreter.

```python
#this should work!
import test_package
```

If you have used the default package settings then you will have installed a package called `pypi-template` (version =0.1).  To uninstall use the package name:

`pip uninstall pypi-template`




