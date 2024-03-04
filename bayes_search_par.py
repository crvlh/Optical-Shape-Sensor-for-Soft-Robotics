# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:26:20 2024

@author: vinic
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Generate the reduced PCA training dataset (axis: x or y[configs_y_train])
X, y = data_train, configs_x_train
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_norm)

# ElasticNet Hyperparameter Optimization

# Define the search space for hyperparameters
search_spaces_elastic_net = {
    'l1_ratio': Real(0.001, 0.999, prior='uniform'),
    'alpha': Real(0.001, 0.999, prior='uniform'),
    'fit_intercept': Categorical([True, False])
}

# Define the evaluation function for Bayesian Optimization
def evaluate_elastic_net(params):
    model = ElasticNet(**params, random_state=42)
    scores = cross_val_score(model, X_pca, y, cv=5, scoring='neg_mean_absolute_error', random_state=0)
    return -scores.mean()

# Create the BayesSearchCV object for hyperparameter optimization
opt_elastic_net = BayesSearchCV(
    ElasticNet(random_state=0),
    search_spaces_elastic_net,
    n_iter=100,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    verbose=1,
    random_state=0
)

# Execute the search for the best hyperparameters
opt_elastic_net.fit(X_pca, y)

# Print the best hyperparameters found
print("Best hyperparameters for ElasticNet: ", opt_elastic_net.best_params_)

# SVR Hyperparameter Optimization

# Define the search space for hyperparameters
search_spaces_svr = {
    'C': Real(0.1, 500, prior='uniform'),
    'epsilon': Real(0.001, 1, prior='uniform'),
    'kernel': Categorical(['poly', 'rbf']),
    'degree': Integer(1, 3, prior='uniform')
}

# Define the evaluation function for Bayesian Optimization
def evaluate_svr(params):
    model = SVR(random_state=42, **params)
    scores = cross_val_score(model, X_pca, y, cv=5, scoring='neg_mean_absolute_error', random_state=0)
    return -scores.mean()

# Create the BayesSearchCV object for hyperparameter optimization
opt_svr = BayesSearchCV(
    SVR(),
    search_spaces_svr,
    n_iter=100,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    verbose=1,
    random_state=0
)

# Execute the search for the best hyperparameters
opt_svr.fit(X_pca, y)

# Print the best hyperparameters found
print("Best hyperparameters for SVR: ", opt_svr.best_params_)
