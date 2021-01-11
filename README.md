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

### Managing the versioning your package

All python packages should behave in this way:

```python
>>> import test_package
>>> print(test_package.__version__)
'0.1'
```
So we need an __init__.py file for our package to reference a version number AND we need to include a version number in the setup script.

As far as I can tell there is no ideal way to set this up. See PEP 396 for discussion

* https://www.python.org/dev/peps/pep-0396/#examples 

For simpler packages like small scientific ones I am aiming to produce I have opted for the following pattern:

1. Keep the version numbering external to the setup script.
2. In the package __init__.py include a `__version__` string attribute and set that to the version number
3. In setup.py I include the following code

```python
#import the version
from test_package import __version__
```

This doesn't include any auto tick up on the version number!

### including data in your package

For data science project you may want to include some example or test data within your package. 

As an example, the package in the template repo includes a subdirectory `test_package/data` containing a single (dummy) CSV.

One thing I learnt **the hard and annoying way** is that data is not included in your python package by default!  

To make sure data is included I took two steps

#### Tell `setup()` that it is expecting data

There are a couple of options here.  The way I have implemented this elsewhere is:

```python
    #Tells setup to look in MANIFEST.in
    include_package_data=True,
```

The above snippet tells the `setup()` function to look in a top level file called `MANIFEST.in`.  This contains a list of files to include in the package:

```
include *.txt
recursive-include test_package/data *.csv
```

As an alternative you can use:

```python
    #can instead of MANIFEST.in
    package_data={"test_package": ["data/*.csv"]},
```

I've found that this only works if `include_package_data=False`

* RTM: https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html 

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




