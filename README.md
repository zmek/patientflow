# PyPi

If you want to deploy a python package professionally, sustainably, and easily to colleagues, clients or just everyone in general then a great way to do it is using the Python Package Index (PyPi) and `pip`.  In this chapter we will learn how to setup a python package so that it is ready to be uploaded to PyPi and also how to use `setup-tools` and `twine`.

## What is pip?

A package management system for installing **local packages and python packages from the Python package index (PyPi - pronounced Pie Pie)**.  Example usage:

`$ pip install numpy`

or

`$ pip install numpy==1.18.0`

I recommend exploring what packages are available on [PiPy](https://pypi.org/).  The chances are when you get stuck in data science project there will be a package on pypi to help you.  For example, if you need to solve some really complex multi-objective optimisation problems you could pip install [DEAP](https://pypi.org/project/deap/).  Obviously you need to have an up-to-date version of `pip` before you can install anything.  If you are using an Anaconda distribution or conda environment  (for example, `hds_code` provided with this book) you should already have it. If you are stuck there are some instructions [here](https://packaging.python.org/tutorials/installing-packages/)

## Why use PyPi for your own projects?

In summary, it is useful if you want to use a package in multiple projects without asking the user to manage the source code (or binaries) themselves. This can, of course, be managed in various ways, but I've found that people I work with have had an easier time when the software is managed by pip.  For example, we haven't manually managed the source for `pandas`, `matplotlib` or `numpy` in this book.  That's dar too complicated `pip` (and other package managers) make the packages accessible to others.  That's a great thing for **open science** and health data science.

So the obvious use case for pip in health data science is if you want others to be able to easily install and use your software/project in their own work.  I use it a lot for educational software for students.  I like the idea that students can use course learning software after they leave a University and access updated versions of it if they want to refresh their skills.  

>  I also recommend making your source code open via a cloud based version control system such as GitHub.  In fact, you can link to this from your PyPi project page.

## Setting up a git repo for a PyPi project.

I don't want to be too prescriptive here.  It really is up to you and your collaborators how you organise your own code.  There are many suggestions online as well. The structure I am going to introduce here is a simple one I use in my own projects.  You will see variations of it online.  Here's the template repo.  We will then take a look at the elements individually.   You can view the template repository and the code it contains on [GitHub](https://github.com/health-data-science-OR/pypi-template)

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

### `environment.yml`

One thing I have learnt from the Open Source community is it is useful to include a virtual conda environment for yourself and developers who contribute to the library.  That is what is in `environment.yml`. One omission in the template repo is that it is useful to **install the latest version of your own package into the dev environment!**  This means you can create example Jupyter notebooks where other data scientists can learn how to use your package.  See for example:  https://github.com/TomMonks/forecast-tools 

```yaml
name: pypi_package_dev
channels:
  - defaults
dependencies:
  - jupyterlab=1.2.6
  - matplotlib=3.1.3
  - numpy=1.18.1
  - pandas=1.0.1
  - pip=20.0.2
  - pytest=5.3.5
  - python=3.8.1
```

### `setup.py`

This is the important file for `pip`.  `setup.py `controls the installation of your package.  I've included a template in the repo.  We need to use the `setuptools` PyPI package to do the installation.  I've included `setuptools` in the dev environment, but you can install it manually:

`pip install setuptools`

Here is what I have included in `setup.py`

```python
import setuptools
from test_package import __version__

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypi-template",
    version=__version__,
    author="Thomas Monks",
    # I've created a specific email account before and forwarded to my own.
    author_email="generic@genericemail.com",
    license="The MIT License (MIT)",
    description="A short, but useful description to appear on pypi",
    # read in from readme.md and will appear on PyPi
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/health-data-science-OR/pypi-template",
    packages=setuptools.find_packages(),
    # if true look in MANIFEST.in for data files to include
    include_package_data=True,
    # 2nd approach to include data is include_package_data=False
    package_data={"test_package": ["data/*.csv"]},
    # these are for documentation 
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    install_requires=requirements,
)
```

### Managing the versioning your package

All python packages should behave in this way:

```python
>>> import test_package
>>> print(test_package.__version__)
'0.1'
```
So we need an __init__.py file for our package to reference a version number AND we need to include a version number in the setup script.

> There is no ideal way to set this up. See PEP 396 for [discussion](https://www.python.org/dev/peps/pep-0396/#examples)

For simpler packages like small scientific ones we are aiming to produce I have opted for the following pattern:

1. Keep the version numbering external to the setup script.
2. In the package __init__.py include a `__version__` string attribute and set that to the version number
3. In setup.py I include the following code

```python
# import the version
from test_package import __version__
```
Note that you need to be careful about version numbering here: It has to be managed manually.  One way to handle that is via some testing before updating a package on pypi.

### Including data in your package

For a data science project you may want to include some example or test data within your package. 

As an example, the package in the template repo includes a subdirectory `test_package/data` containing a single (dummy) CSV.

One thing I learnt **the hard and annoying way** is that data is not included in your python package by default!  

To make sure data is included you can take two steps:

#### Tell `setup()` that it is expecting data

There are a couple of options here.  Here's the firs:

```python
    # Tells setup to look in MANIFEST.in
    include_package_data=True,
```

The above snippet tells the `setup()` function to look in a top level file called `MANIFEST.in`.  This contains a list of files to include in the package:

```
include *.txt
recursive-include test_package/data *.csv
```

As an alternative you can use:

```python
    # use instead of MANIFEST.in
    package_data={"test_package": ["data/*.csv"]},
```

Note that the second way I've described requires you to set `include_package_data=False`

* Some extra help can be found here in the setup tools manual: https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html 

### Local installation and uninstallation of your package

Now that we have a `setup.py` and have installed `setuptools` we can use them to install our package locally!  Navigate to the repo on your local machine and run the command below.

`pip install .`

**Exercise**: Test out your install by launching the python interpreter.

```python
# this should work!
import test_package
print(test_package.__version__)
```

If you have used the default package settings then you will have installed a package called `pypi-template` (version = 0.1).  To uninstall use the package name:

`pip uninstall pypi-template`

## Publishing your package on PyPI

The first thing to say is that **there is a PyPI test site!**  This is incredibly helpful.  I found that I made mistakes the first couple of times I tried to get this to work e.g. no data was included in my package.  Once it is published on the testpypi site you can install it!

You need to go to https://testpypi.python.org and create an account.  

### Including source and wheel distributions

It is recommended that you include both a **source** and **wheel** distribution in your python package on pypi. Wheels are an advanced topic and it took me a while to get my head around what a wheel distribution actually is and why it is useful!  A nice phrase I came across is that '*wheels make things go faster*'

* A source is just what you think it is.  It is your source code!

* A wheel (.whl) is a ready-to-install python package i.e. no build stage is required. This is very useful for example if you have written custom C or Rust extensions for python. The wheel contains your extensions compiled and hence skips the build step.  Wheel's therefore make installation more efficient.  

> You can create universal wheel's i.e. applicable to both python 2 and 3 or pure python wheels i.e. applicable to only python 3 or python 2, but not both.

To install wheel use

```bash
$ pip install wheel
```

To produce the source and the wheel run 

```bash
$ python setup.py sdist bdist_wheel
```

* Now take a look in ./dist.  You should see both a zip file containing the source code and a .whl file containing the wheel.

A .whl file is named as follows:

```
{dist}-{version}-{build}-{python}-{abi}-{platform}.whl
```

For additional information on wheels I recommend checking out [https://realpython.com/python-wheels/](https://realpython.com/python-wheels/)

### Using twine and testpypi

To publish on pypi and testpypi we need to install an simple to use piece of software called `twine`

```bash
$ pip install twine
```

It is sensible to check for any errors before publishing!  To do this run the code below

```bash
$ twine check dist/*
```
This will check your source and wheel distributions for errors and report any issues.

> Before you push to testpypi you need to have a unique name for your package!  I recommend making up your own name, but if you are feeling particularly unimaginative then use `pypi_template_{random_number}`. Set this in `setup.py`. **Check if it has been created and exists on testpypi first!**

To publish on testpypi is simple.  The code below will prompt you for your username and password.

```bash 
$ twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

If this uploads successfully you can then `pip` install from testpypi as follows.  The URL for your package is available on the project page for testpypi.

```bash
$ pip install -i https://test.pypi.org/simple/{your_package_name}==0.1.0
```

### Publish on pypi production

First I just want to say that you shouldn't really publish on the main production pypi site for unless you need to.  Use it when necessary to help your own research, work or colleagues, but not for testing purposes: use testpypi instead.  **You will need a separate account for PyPI.**.  If you intend to publish to pypi I recommend searching the index first in order to identify any potential name clashes.  

When you are ready you can upload use `twine`

```bash
$ twine upload dist/*
```

## Building publishing into your workflow with GitHub Actions

The manual steps I've outlined here are somewhat historical.  Most modern projects make use of version control in the cloud such as GitLab or GitHub.  These include ways to automatically publish updates to pypi.  One such way is with GitHub actions.  For example, I use **the publish to pypi action** when code is merge into the 'main' branch.

To set this up you will need to supply GitHub with a username and password for pypi.  Its stored securely, but you may rightly have concerns about privacy and security. My approach has been to create a secondary maintainer account for pypi rather than storing my main PyPI credentials to GitHub. I will leave it up to you to make a decision about if you feel this is necessary.  It can always be updated at a later date.

You can read more about Github actions [here](https://docs.github.com/en/actions)

