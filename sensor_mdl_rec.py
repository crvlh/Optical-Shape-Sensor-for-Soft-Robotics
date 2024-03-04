# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:27:51 2024

@author: vinic
"""
import pandas as pd
import numpy as np
import random
import glob
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from scipy.interpolate import splprep, splev
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from tensorflow.keras import regularizers
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statistics

random.seed(0)
########## START OF DATA IMPORT ##########
n_wavelength_effective = 1063  # number of wavelengths
skip_rows = 1062  # row of the first considered wavelength
n_components = 9  # number of PCA components

# Import data

# Import structure deformation configurations
data_folder = "C:\\Users\\"  # SHAPE CONFIGURATIONS DIRECTORY
config_file_name = "shape_configs.xlsx"
df = pd.read_excel(os.path.join(data_folder, config_file_name), header=None)
num_samples = df.shape[0]
last_col = (num_samples * 2 + 1)
configs_x_train = np.array(df.iloc[:num_samples, 1].tolist())  
configs_y_train = np.array(df.iloc[:num_samples, 2].tolist())  

# Import files (multiple files, each file a sample), read spectral data, filter wavelengths of interest.
data_folder_trainval = "C:\\Users\\"  # SPECTRA SAMPLES DIRECTORY
file_list = glob.glob(os.path.join(data_folder_trainval, "*.txt"))
file_list = sorted(file_list, key=lambda x: int(os.path.basename(x).split(".")[0]))
data_trainval = [pd.read_csv(file, delimiter='\t', skiprows=skip_rows, nrows=n_wavelength_effective, usecols=range(1, 2))
                  for file in file_list]
data_trainval = np.reshape(np.array(data_trainval), (num_samples, n_wavelength_effective))
wavelengths = pd.read_csv(file_list[0], sep='\t', header=None, usecols=[0], skiprows=(skip_rows + 1),
                          nrows=n_wavelength_effective, index_col=None)

# Organize into training/validation and test sets
test_size = 0.3
data_train, data_test, configs_x_train, configs_x_test = train_test_split(data_trainval, configs_x_train,
                                                                          test_size=test_size, random_state=0)
data_train, data_test, configs_y_train, configs_y_test = train_test_split(data_trainval, configs_y_train,
                                                                          test_size=test_size, random_state=0)
########## END OF DATA IMPORT ##########

# Plot spectral data of training/validation and test sets

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Spectral data of training and validation
ax1.plot(wavelengths, np.transpose(data_train), linewidth=0.1)
ax1.tick_params(direction='in')
ax1.set_title("Training and validation data")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Relative Transmitted Intensity(%)")
ax1.set_xlim(475, 750)
ax1.set_ylim(80, 120)

# Spectral data of test set
ax2.plot(wavelengths, np.transpose(data_test), linewidth=0.1)
ax2.tick_params(direction='in')
ax2.set_title("Test data")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Relative Transmitted Intensity (%)")
ax2.set_xlim(475, 750)
ax2.set_ylim(80, 120)

plt.show()

# Create pipelines

enet = ElasticNet()
pipe_enet = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0), enet)

svr = SVR()
pipe_svr = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0), svr)

# Define grid search parameters

parameters_enet = {
    'elasticnet__l1_ratio': [0.001, 0.7107431537091249],
    'elasticnet__alpha': [0.267727067383456, 0.23059800961692473],
    'elasticnet__fit_intercept': [True]
}

parameters_svr = {
    'svr__C': [115.7556512388615, 62.44131262881692],
    'svr__epsilon': [0.11744208254116344, 0.3093274857712457],
    'svr__kernel': ['rbf'],
    'svr__degree': [2, 1]
}

# Train predictive regression models with optimized parameters

grid_enet_x = GridSearchCV(pipe_enet, parameters_enet, cv=10, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_enet_x.fit(data_train, configs_x_train)
grid_enet_y = GridSearchCV(pipe_enet, parameters_enet, cv=10, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_enet_y.fit(data_train, configs_y_train)

grid_svr_x = GridSearchCV(pipe_svr, parameters_svr, cv=10, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_svr_x.fit(data_train, configs_x_train)
grid_svr_y = GridSearchCV(pipe_svr, parameters_svr, cv=10, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_svr_y.fit(data_train, configs_y_train)

# Print overall results for each model
print(" VALIDATION RESULTS OF REGRESSION MODELS FOR X-AXIS POSITION PREDICTION:")

print("\n Elastic Net Results")
print("Best estimators:", grid_enet_x.best_estimator_)
print("Best score obtained (MAE): %.4f" % (grid_enet_x.best_score_ * -1))
print("Best parameters:", grid_enet_x.best_params_)

print("\n SVR Results")
print("Best estimators", grid_svr_x.best_estimator_)
print("Best score (MAE): %.4f" % (grid_svr_x.best_score_ * -1))
print("Best parameters:", grid_svr_x.best_params_)

print("\n VALIDATION RESULTS OF REGRESSION MODELS FOR Y-AXIS POSITION PREDICTION:")

print("\n Elastic Net Results")
print("Best estimators:", grid_enet_y.best_estimator_)
print("Best score (MAE): %.4f" % (grid_enet_y.best_score_ * -1))
print("Best parameters:", grid_enet_y.best_params_)

print("\n SVR Results")
print("Best estimators", grid_svr_y.best_estimator_)
print("Best score (MAE): %.4f" % (grid_svr_y.best_score_ * -1))
print("Best parameters:", grid_svr_y.best_params_)

# Use the best models (optimized by k-fold cross-validation) to make predictions on the test set for x-axis position

pred_x_enet_test = grid_enet_x.predict(data_test)
pred_x_svr_test = grid_svr_x.predict(data_test)

# Use the best models (optimized by k-fold cross-validation) to make predictions on the test set for y-axis position

pred_y_enet_test = grid_enet_y.predict(data_test)
pred_y_svr_test = grid_svr_y.predict(data_test)

# Evaluate test performances for x-axis prediction. Preliminary version with treval set instead of test set (replace)

mae_enet_x = mean_absolute_error(configs_x_test, pred_x_enet_test)
mae_svr_x = mean_absolute_error(configs_x_test, pred_x_svr_test)

print("\n TEST RESULTS OF REGRESSION MODELS FOR X-AXIS PREDICTION:")

print("MAE ENET: %.4f" % mae_enet_x)
print("MAE SVR: %.4f" % mae_svr_x)

# Evaluate test performances for y-axis prediction

mae_enet_y = mean_absolute_error(configs_y_test, pred_y_enet_test)
mae_svr_y = mean_absolute_error(configs_y_test, pred_y_svr_test)

print("\n TEST RESULTS OF REGRESSION MODELS FOR Y-AXIS PREDICTION:")

print("MAE ENET: %.4f" % mae_enet_y)
print("MAE SVR: %.4f" % mae_svr_y)

# Reconstruct the shape of the flexible structure with specific model
# Compare predictions and targets of tests samples

samples_test_random = random.sample(list(range(0, configs_x_test.shape[0])), configs_x_test.shape[0])

for i in range(0, configs_x_test.shape[0]):
    posx_p = pred_x_svr_test[samples_test_random[i]]
    posy_p = pred_y_svr_test[samples_test_random[i]]
    x = [0, 0, posx_p]
    y = [0, 0, posy_p]
    z = [20, 10, 0]

    # Use splprep() to smooth the curves of the line with a quadratic spline curve
    tck, u = splprep([x, y, z], s=0.0, k=2)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new, z_new = splev(u_new, tck)

    posx_p_real = configs_x_test[samples_test_random[i]]
    posy_p_real = configs_y_test[samples_test_random[i]]
    x_real = [0, 0, posx_p_real]
    y_real = [0, 0, posy_p_real]
    z_real = [20, 10, 0]

    tck_real, u_real = splprep([x_real, y_real, z_real], s=0.0, k=2)
    u_new_real = np.linspace(u_real.min(), u_real.max(), 1000)
    x_new_real, y_new_real, z_new_real = splev(u_new_real, tck_real)

    num_circles = 50
    dist_circles = (u.max() - u.min()) / num_circles

    x_circles, y_circles, z_circles = splev(np.arange(u.min(), u.max(), dist_circles), tck)
    x_circles_real, y_circles_real, z_circles_real = splev(np.arange(u_real.min(), u_real.max(), dist_circles),
                                                           tck_real)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_new, y_new, z_new, color='b')
    ax.plot(x_new_real, y_new_real, z_new_real, color='r')

    for i in range(len(x_circles)):
        u_circle = np.linspace(0, 2 * np.pi, 100)
        x_circle = x_circles[i] + np.cos(u_circle)
        y_circle = y_circles[i] + np.sin(u_circle)
        z_circle = z_circles[i] * np.ones_like(u_circle)
        ax.plot(x_circle, y_circle, z_circle, color='b', alpha=0.4)

    for i in range(len(x_circles)):
        u_circle_real = np.linspace(0, 2 * np.pi, 100)
        x_circle_real = x_circles_real[i] + np.cos(u_circle_real)
        y_circle_real = y_circles_real[i] + np.sin(u_circle_real)
        z_circle_real = z_circles_real[i] * np.ones_like(u_circle_real)
        ax.plot(x_circle_real, y_circle_real, z_circle_real, color='r', alpha=0.4)

    ax.set_xlim([10, -10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 21])

    red_patch = mpatches.Patch(color='red', label='Actual shape')
    blue_patch = mpatches.Patch(color='b', label='Detected shape')
    ax.legend(handles=[red_patch, blue_patch], loc='best')

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')

    ax.view_init(elev=20, azim=89.999999999999)

plt.show()

# Calculate the standard deviation of absolute test sample errors for a specific model

std_dev_errors_x_test = statistics.stdev(abs(pred_x_svr_test - configs_x_test))
std_dev_errors_y_test = statistics.stdev(abs(pred_y_svr_test - configs_y_test))

print("\n STANDARD DEVIATION OF ABSOLUTE ERRORS X: %.4f" % std_dev_errors_x_test)
print("STANDARD DEVIATION OF ABSOLUTE ERRORS Y: %.4f" % std_dev_errors_y_test)