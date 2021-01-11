# pypi-template

Notes and a template repo for PyPi Projects.

T.Monks. Jan 2021.


## Useful posts for learning:
* https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/#what-is-pip

* https://nrempel.com/blog/how-publish-python-package-pypi-comprehensive-guide/

* https://realpython.com/python-wheels/

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
print(test_package.__version__)
```

If you have used the default package settings then you will have installed a package called `pypi-template` (version =0.1).  To uninstall use the package name:

`pip uninstall pypi-template`

## Publishing your package on PyPI

The first thing to say is that there is a PyPI test site!  This was incredibly helpful as I found that I made mistakes the first couple of times I tried to get this to work e.g. no data was included in my package.  Once it is published on the testpypi site you can install it!

You need to go to https://testpypi.python.org and create an account.  

### Including source and wheel distributions

It is recommended that you include both a **source** and **wheel** distribution in your python package.  

> It took me a while to get my head around what a wheel distribution actually is and why it is useful!  A nice phrase I came across is that '*wheels make things go faster*' I'm yet to fully master them and particularly need to research platform specific wheels.  

This site has a nice explanation: https://realpython.com/python-wheels/

* A source is just what you think it is.  It is your source code!

* A wheel (.whl) is a ready-to-install python package i.e. no build stage is required. This is very useful for example if you have written custom C extensions for python. The wheel contains your extensions compiled and hence skips the build step.  Wheel's therefore make installation more efficient.  

> You can create universal wheel's i.e. applicable to both python 2 and 3 or pure python wheels i.e. applicable to only python 3 or python 2, but not both.

To install wheel use

```
pip install wheel
```

To produce the source and the wheel run 

```
python setup.py sdist bdist_wheel
```

* Now take a look in ./dist.  You should see both a zip file containing the source code and a .whl file containing the wheel.

A .whl file is named as follows:

```
{dist}-{version}-{build}-{python}-{abi}-{platform}.whl
```

### Using twine and testpypi

To publish on pypi and testpypi we need to install `twine`

```
pip install twine
```

It is sensible to check for any errors before publishing!  To do this run the code below

```
twine check dist/*
```
This will check your source and wheel distributions for errors and report any issues.

> Before we push to testpypi you need to have a unique name for your package! For this tutorial I recommend 'pypi_template_{random_number}. Set this in setup.py. Check if it has been created and exists on testpypi first!

Let's push to testpypi.  The code below will prompt you for your username and password.

```
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

```
Pip install from testpypi as follows:

```
pip install -i https://test.pypi.org/simple/ pypi-template-2222==0.1.0
```


### Publish on full fat pypi

You will need a seperate account for PyPI.  (Let's not publish our test template there!)

```
twine upload dist/*
```

# Building publishing into your workflow with GitHub Actions

* Github now provides template actions
* I use the publish to PyPI action when code is merge into the 'main' branch.
* For forecast-tools I created a maintainer account for PyPI called `forecast-tools-admin` rather than giving my main PyPI credentials to GitHub.  I'm not sure if this matters in practice or not.


https://pypi.org/project/forecast-tools/#description