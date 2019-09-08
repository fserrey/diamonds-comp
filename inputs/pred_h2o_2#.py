# Packages to be imported

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import h2o # Package for ML model
from h2o.estimators.gbm import H2OGradientBoostingEstimator


import os
print(os.listdir("../input"))
h2o.init() # First, we initiate h2o to work with it

# Data loading
data_path = "../input/diamonds_train.csv"
data_path_test = "../input/diamonds_test.csv"

# Tunning variables
train["vol"] = train["x"]*train["y"]*train["z"] # Creating a new one based on "volume"
test["vol"] = test["x"]*test["y"]*test["z"]

train["carat-col"]=train["carat"] * train["vol"]
test["carat-col"]=test["carat"] * test["vol"]

# Set train-test dataframes
train = train[['carat-col', 'cut', 'color', 'clarity', 'depth', 'table', 'price']]
test = test[['carat-col', 'cut', 'color', 'clarity', 'depth', 'table']]

ord_cut = {
    "Premium":4, "Very Good": 3,"Ideal":2,"Good":1, "Fair":0,  
    "G":4,"E":6,"F":5,"H":3,"D":7,"I":3,"J":0,
    "SI1":3,"VS2":4,"SI2":2,"VS1":5,"VVS2":6,"VVS1":7,"IF":8,"I1":1     
    }

train.replace(ord_cut, inplace=True)
test.replace(ord_cut, inplace=True)

# Using h2o syntax, we import the datasets as h2o 
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)

# Set train-test dataframes
y = "price" 
x = train.columns
x.remove(y)

train_final, valid = train.split_frame(ratios=[0.8]) # We make the split of training and validation

modelo = H2OGradientBoostingEstimator(
    ntrees=440,
    learn_rate=0.5531490180631663,
    max_depth=10,
    #sample_rate: 0.6117256495829282,
    model_id="gbm_covType_v2",
    #min_rows:16,
    seed=2000000
)

# Prediction
final_pred = modelo.predict(test_data=test)

# Performance
perf_valid = aml.leader.model_performance(final_pred)
print(perf_train)
print(perf_valid)


# Setup prediction as dataframe
pred_pd = pred.as_data_frame()
sub = sub_data.as_data_frame() #We use this dataframe as structure

# Final setup and saved as .csv
sub['price'] = pred_pd
h2o_test = sub.copy()
h2o_test = pd.DataFrame(h2o_test.price)
h2o_test.index.name='id'
h2o_test.to_csv("h2o_predictions.csv")