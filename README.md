## Optical Fiber Macro-Bend Sensor for Shape Monitoring in Flexible Structures

![abstract_figure3](https://github.com/crvlh/Optical-Shape-Sensor-for-Soft-Robotics/assets/120674953/be5daff3-767d-42b6-8acc-b1df7b38dae6)

### Research paper 
Optical Fiber Macro-Bend Sensor for Shape Monitoring in Flexible Structures

### Overview
This repository presents a fiber optic sensor-based approach for monitoring deformations in flexible structures, with a focus on soft robotic systems. The sensor utilizes macrobend-induced variations in optical intensity to track structural deformations, enabling real-time shape sensing. The methodology involves preprocessing optical transmission spectra using dimensionality reduction techniques, training predictive models through Bayesian parameter optimization, and evaluating model performance for shape monitoring.

### Files attached to this repository
- bayes_search_par: Bayesian parameter optimization.

* pca_red: Principal component analysis (PCA) preprocessing.

+ data_shape_configs_spectrasamples.rar: Comprises spectral data (.txt files) and deformation configurations (.xlsx file).

- sensor_mdl.red: Contains grid search implementation for model training and parameter selection, producing reconstructions on the test set and presenting errors.

- test_predictions.mp4: Provides three-dimensional reconstructions of test samples in video format.

### Methodology
The methodology involves preprocessing optical transmission spectra using PCA to reduce computational complexity. Bayesian parameter optimization is employed for training predictive models, followed by a grid search to select optimal hyperparameters. This ensures efficient shape monitoring while maintaining computational simplicity.

### Results
The results demonstrate the effectiveness of the proposed sensor in monitoring deformations, with average detection errors of less than 1.0 cm. Three-dimensional reconstructions confirm the sensor's accuracy in tracking real structural deformations. The repository includes visualizations of reconstructions and performance metrics for model evaluation.

### Usage
Users can utilize the provided files to implement the proposed methodology for shape monitoring in flexible structures. They are encouraged to explore and experiment with new models, optimization techniques, and result analysis methods using the repository as a framework. Detailed instructions and documentation are included to facilitate customization and experimentation.

### Acknowledgment
Lablaser - Federal University of Technology – Parana (UTFPR)

For any additional questions or required files, users can post them in the "Issues" section, where they will be promptly addressed.
