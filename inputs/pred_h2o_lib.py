# Python 3 analytics libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import os
import h2o
from h2o.automl import H2OAutoML

# Inisialitation of H2O lib and data loading
h2o.init() 
data_path = "../input/diamonds_train.csv"
data_path_test = "../input/diamonds_test.csv"

# Var definition (data loading into h2o lib)
train = h2o.import_file(data_path)
df_test = h2o.import_file(data_path_test)
sub_data = h2o.import_file("../input/sample_submission.csv") # We load this file to use it as a model to save the final prediction

y = "price"
x = train.columns
x.remove(y)

run_automl_for_seconds = 7200 # Time setup for AutoML model
aml = H2OAutoML(max_runtime_secs =run_automl_for_seconds)

# Train
train_final, valid = train.split_frame(ratios=[0.9])
aml.train(x = x, y = y, training_frame=train_final, validation_frame=valid)

# Prediction
leader_model = aml.leader
pred = leader_model.predict(test_data=df_test)

# Performance evaluation
perf = aml.leader.model_performance(valid)
print(perf) 

# Setup prediction as dataframe
pred_pd = pred.as_data_frame()
sub = sub_data.as_data_frame() #We use this dataframe as structure

# Final setup and saved as .csv
sub['price'] = pred_pd
h2o_test = sub.copy()
h2o_test = pd.DataFrame(h2o_test.price)
h2o_test.index.name='id'
h2o_test.to_csv("h2o_predictions.csv")