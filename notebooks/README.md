# About the notebooks

This folder contains a series of notebooks that demonstrate the process of modeling emergency demand in healthcare through a structured approach. Notebooks combine commentary with code and the results produced by that code. Here’s how different audiences can benefit from these notebooks:

- **For non-programmers seeking to understand the approach:** If you're not familiar with programming, you can read the narrative sections of the notebooks as if it were a blog post to understand the strategies employed, and skip the code snippets.
- **For those new to Python or looking to learn:** If you have some coding experience but are new to Python, the code snippets, combined with the narrative, provide insights into how Python is applied in modelling emergency bed demand. They show how tasks such as data exploration, model fitting, and result evaluation can be done in Python. Observing the output will help you see the practical outcomes of the modeling.
- **For those interested in using the 'patientflow' package:** These notebooks serve as a guide on how to use the 'patientflow' package. Through walkthroughs of the functions and their applications, you can learn how to integrate this package into your own projects.

## Outline of the notebooks

- **1_Introduce_our_users:** Talks about the users of emergency demand predictions in acute hospitals.
- **2_Modelling_requirements:** Explains design choices that were made to develop a practical model, and shows an example of the output that is sent five times a day at UCLH.
- **3_Introduce_the_datasets:** Introduces the two datasets created to accompany this repository
- **4a_Predict_probability_of_admission_from_ED:** Shows how to train a machine learning model to predict a patient's probability of admission using patient data from the Emergency Department (ED). This includes dividing the data into training, validation, and testing sets, as well as into subsets based on the time of day the predictions are made, applying an XGBoost model for predictions, and saving the models for future use.
- **4b_Predict_demand_from_patients_in_ED** Illustrates how to convert individual admission probabilities into a prediction of overall bed demand
- **4c_Predict_probablity_of_admission_to_specialty:** Shows how to train a model predicting specialty of admission; a sequence of consultation requests is mapped to a probability of being admitted to one of three specialties: medical, surgical, and haematology/oncology, with paediatric patients (under 18) handled differently
- **4d_Predict_demand_from_patients_yet_to_arrive:** Show the use of a time-varying weighted Poisson distribution to predict a number of patients yet to arrive to the ED within a prediction window (say 8 hours) of the time of prediction, by specialty. Demonstrates the use of a function that will take ED performance targets into account when predicting the number admitted by the end of the prediction window
- **5_Model_evaluation**: Discusses how to evaluate the models' predictions
- **6 Bring it all together:** Shows an example of doing live inference using the models trained in the previous steps
- **7 Modelling to show demand:** Shows an example of doing live inference using the models trained in the previous steps

## Preparing your notebook environment

The `PROJECT_ROOT` environment variable needs to be set so notebooks can find the repository root directory. You have various options: -**use a virtual enviromment and set it up within that (recommended)** -**set it temporarily within each notebook**
**set it globally on your computer**

### To set the PROJECT_ROOT environment variable within your virtual environment (recommended)

**Conda environments**

Add PROJECT_ROOT to the `environment.yml` file:

```yaml
variables:
  PROJECT_ROOT: /path/to/patientflow
```

**venv**

Add PROJECT_ROOT to the venv activation script:

```sh
echo 'export PROJECT_ROOT=/path/to/patientflow' >> venv/bin/activate  # Linux/Mac
echo 'set PROJECT_ROOT=/path/to/patientflow' >> venv/Scripts/activate.bat  # Windows
```

The environment variable will be set whenever you activate the virtual environment and unset when you deactivate it.
Replace /path/to/patientflow with your repository path.

### To set the PROJECT_ROOT environment variable temporarily

From the terminal before launching Jupyter:

```sh
Linux/Mac:
export PROJECT_ROOT=/path/to/patientflow

Windows:
set PROJECT_ROOT=C:\path\to\patientflow
```

From a notebook cell:

**Linux/Mac:**

```sh
%env PROJECT_ROOT=/path/to/patientflow
```

Windows:

```sh
%env PROJECT_ROOT=C:\path\to\patientflow
```

Replace /path/to/patientflow with the actual path to your cloned repository.

### To set PROJECT_ROOT environment variable permanently on your system

**Linux/Mac:**
Add to ~/.bashrc or ~/.zshrc:
`export PROJECT_ROOT=/path/to/patientflow`

**Windows:**

```sh
Open System Properties > Advanced > Environment Variables
Under User Variables, click New
Variable name: PROJECT_ROOT
Variable value: C:\path\to\patientflow
Click OK
```

Replace /path/to/patientflow with your repository path. Restart your terminal/IDE after setting.
