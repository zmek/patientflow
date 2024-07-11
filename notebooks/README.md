# About the notebooks

We are developing a series of notebooks that demonstrate the process of modeling emergency demand in healthcare through a structured approach. Notebooks are an effective tool for merging explanatory narratives with practical code and the results produced by that code. Here’s how different audiences can benefit from these notebooks:

- **For non-programmers seeking to understand the approach:** If you're not familiar with programming, you can read the narrative sections of the notebooks like a blog post to understand the strategies employed. Feel free to skip the code snippets.
- **For those new to Python or looking to learn:** If you have some coding experience but are new to Python, the code snippets, combined with the narrative, provide insights into how Python is applied in modeling emergency bed demand. This includes tasks such as data exploration, model fitting, and result evaluation. Observing the output will help you see the practical outcomes of the modeling.
- **For users who prefer not to set up their own coding environment:** Each notebook is designed to run using a provided dataset, aiming to replicate our results. The dataset will hopefully be based on real data from University College London Hospital; we are currently working on a Data Protection Impact Assessment (DPIA) with them. We intend that it will be possible to run these notebooks on Colab and/or BinderHub
- **For those interested in using the 'patientflow' package:** These notebooks serve as a guide on how to use the 'patientflow' package. Through detailed walkthroughs of the functions and their applications, you can learn how to integrate this package into your own projects.

## Outline of the notebooks

Our plan is to create the following notebooks:

- **1 Introducing our users:** Talk about the users of emergency demand predictions, and what their requirements are from a model. View the notebook [here](/notebooks/1%20Introducing%20our%20users.ipynb)
- **2 Introducing emergency demand and its modelling:** Explain design choices made to develop a practical model and show an example of the output that we send at UCLH. View the notebook [here](/notebooks/2%20Introducing%20emergency%20demand%20and%20its%20modelling.ipynb)
- **3 Introducing the datasets:** Introduce the three datasets we have created to accompany this repository
- **4 Modelling probability of admission from ED:** Show how to train a machine learning model to forecast admission likelihood based on patient data from the Emergency Department (ED). This includes dividing the data into training, validation, and testing sets, as well as into subsets based on the time of day the predictions are made, applying an XGBoost model for predictions, evaluating and interpreting these predictions, and saving the models for future use.
- **5 Turning individual probabilities into bed counts** Illustrate how to convert individual admission probabilities into an overall bed demand forecast.
- **6 Modelling probability of admission to specialty if admitted:** Show that a rooted directed tree can be used to represent a sequence of consultation requests, which in turn can be mapped to a probability of being admitted to one of three specialties: medical, surgical, haematology/oncology
- **7 Turning individual predictions into bed counts by specialty** Combine the output from previous steps into a probability distribution for each specialty. Break down the predictions by different categories (male/female).
- **8 Modelling patients yet to arrive:** Show the use of a time-varying Poisson distribution to predict a number of patients yet to arrive with a prediction window (say 8 hours) of the time of prediction, by specialty. Demonstrate the use of a function that will take ED performance targets into account when predicting the number admitted by the end of the prediction window
- **9 Evaluating the predictions**: Discuss the importance of evaluating predictions and talk about what it means to evaluate a model of aspirational predictions
- **10 Modelling using only OPEL data**: Show how you might use a historical dataset, plus real-time information that is sent to your Integrated Care Board for OPEL reporting, to make demand predictions, and evaluate the loss of predictive power compared with having more data
- **11 Putting it all together:** Show an example of doing live inference using these models
