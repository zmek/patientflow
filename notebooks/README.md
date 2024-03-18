# About the notebooks

The notebooks showing how to model emergency demand in a series of steps. Each can be run against a provided dataset, which should replicate the same results. 

Eventually, it will be possible to run these on Colab and/or BinderHub, so that you do not have to create your own environment to run them. 

## Outline of the notebooks

Our current plan is to create the following notebooks:

- **Introducing our users:** Talk about the users of emergency demand predictions, and what their requirements are from a model
- **Introducing emergency demand and its modelling:** Explain design decisions that we made in order to produce a useful model and show an example of the output that we send at UCLH
- **Introducing the datasets:** Introduce the three datasets we have created to accompany this repo
- **Modelling probability of admission from ED:** Show how to train a Machine Learning model on a dataset that represents patients while they were in the ED. Separate the data into train, validation and test sets, and into a different set for each time of day at which the predictions will be created, run a XGBoost model to predict each patient's probability of admission at those time, evaluate and interpret the results and save the models for later
- **Turning individual probabilities into bed counts** Show how to create a probability distribution over a number of beds. Evaluate and interpret the results
- **Modelling probability of admission to specialty if admitted:** Show that a rooted directed tree can be used to represent a sequence of consult requests, which in turn can be mapped to a probability of being admitted to one of three specialties: medical, surgical, haematology/oncology
- **Turning individual predictions into bed counts by specialty** Combine the output from previous steps into a probability distribution for each specialty. Break down the predictions by different categories (male/female). 
- **Modelling patients yet to arrive:** Show the use of a time-varying Poisson regression to predict a number of patients yet to arrive with a time window (say 8 hours) of the time of prediction, by specialty. Demonstrate the use of a function that will take ED performance targets into account when predicting the number admitted by the end of the time window
- **Putting it all together:** Show an example of doing live inference using these models