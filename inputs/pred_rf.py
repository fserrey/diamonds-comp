# Packages to be imported

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import statsmodels as stat
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 


diamonds_train = pd.read_csv("../input/diamonds_train.csv")
diamonds_test = pd.read_csv("../input/diamonds_test.csv")

# Tunning variables - Label Encoding for train_csv
diamonds_encode = diamonds_train.copy()
diamonds_encode.columns
labelencoder = LabelEncoder()
diamonds_encode["cut"] = labelencoder.fit_transform(diamonds_encode.cut)
diamonds_encode["color"] = labelencoder.fit_transform(diamonds_encode.color)
diamonds_encode["clarity"] = labelencoder.fit_transform(diamonds_encode.clarity)

train_y = diamonds_encode.price
diamonds_y_test = diamonds_encode_test[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
diamond_x_train = diamonds_encode[[x for x in diamonds_encode.columns if x != 'price']]

# Scaling
scaler = RobustScaler()

scaler_y_train = scaler.fit_transform(diamond_x_train)
diamonds_y_test = scaler.fit_transform(diamonds_y_test)

# Grid searching and Random Forest train

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}# Create a based model
rf = RandomForestRegressor()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(diamond_x_train, scaler_y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(criterion='mse', n_estimators = 1500, max_depth=4, random_state = 42)

# Train - Predict
rf.fit(scaler_train, train_y)
y_pred = rf.predict(scaler_test)

# Setup prediction as dataframe
forest_predictions = pd.DataFrame(y_pred)
forest_predictions.index.name='id'
forest_predictions.columns = ["price"]

forest_predictions.to_csv("forest_predictions_robustscaler.csv")