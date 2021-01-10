import setuptools

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypi-template",
    #there must be an way to auto tick up the version number...
    version="0.1",
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
    #include the below if you want to include data in the package...
    include_package_data=True,
    package_data={"": ["pypi_template/data/*.csv"]},
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
