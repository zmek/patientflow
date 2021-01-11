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
    name="pypi-template_2222",
    #there must be an way to auto tick up the version number...
    version=__version__,
    author="Thomas Monks",
    #I've created a specific email account before and forwarded to my own.
    author_email="generic@genericemail.com",
    license="The MIT License (MIT)",
    description="A short, but useful description to appear on pypi",
    #read in from readme.md and will appear on PyPi
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/pypi-template",
    packages=setuptools.find_packages(),
    #if true look in MANIFEST.in for data files to include
    include_package_data=True,
    #2nd approach to include data is include_package_data=False
    package_data={"test_package": ["data/*.csv"]},
    #these are for documentation 
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
