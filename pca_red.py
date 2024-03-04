from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(training_data)

# Apply Principal Component Analysis (PCA)
pca = PCA(n_components=178)
X_pca = pca.fit_transform(X_norm)

# Calculate the proportion of variance explained by each principal component
prop_var = pca.explained_variance_ratio_

# Calculate the cumulative variance explained by each principal component
cum_var = np.cumsum(prop_var)

# Plot the proportion of variance explained by each principal component
plt.bar(range(1, pca.n_components_ + 1), prop_var, label='Individual explained variance')
plt.xlabel('Principal Component')
plt.ylabel('Relative Proportion (%)')
plt.ylim([0, 1])
plt.legend(bbox_to_anchor=(0.98, 0.8), frameon=False)

# Plot the cumulative variance explained by each principal component
plt.twinx()
plt.plot(range(1, pca.n_components_ + 1), cum_var, 'r.-', label='Cumulative explained variance')
plt.ylabel('')
plt.ylim([0, 1])
plt.legend(bbox_to_anchor=(0.4, 0.90), frameon=False)

# Display the plot
plt.show()