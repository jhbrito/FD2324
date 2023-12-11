# Clustering
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture

np.random.seed(1)


def make_samples(n_samples_per_class,
                 class0_x0_mean, class0_x0_std, class0_x1_mean, class0_x1_std,
                 class1_x0_mean, class1_x0_std, class1_x1_mean, class1_x1_std,
                 class2_x0_mean, class2_x0_std, class2_x1_mean, class2_x1_std):
    X = np.zeros((3 * n_samples_per_class, 2), dtype=float)
    Y = np.zeros((3 * n_samples_per_class), dtype=int)

    X[0:n_samples_per_class, 0] = np.random.randn(n_samples_per_class) * class0_x0_std + class0_x0_mean
    X[0:n_samples_per_class, 1] = np.random.randn(n_samples_per_class) * class0_x1_std + class0_x1_mean
    X[n_samples_per_class:2 * n_samples_per_class, 0] = np.random.randn(
        n_samples_per_class) * class1_x0_std + class1_x0_mean
    X[n_samples_per_class:2 * n_samples_per_class, 1] = np.random.randn(
        n_samples_per_class) * class1_x1_std + class1_x1_mean
    X[2 * n_samples_per_class:3 * n_samples_per_class, 0] = np.random.randn(
        n_samples_per_class) * class2_x0_std + class2_x0_mean
    X[2 * n_samples_per_class:3 * n_samples_per_class, 1] = np.random.randn(
        n_samples_per_class) * class2_x1_std + class2_x1_mean

    Y[n_samples_per_class:2 * n_samples_per_class] = np.ones((n_samples_per_class), dtype=float)
    Y[2 * n_samples_per_class:3 * n_samples_per_class] = 2 * np.ones((n_samples_per_class), dtype=float)
    return X, Y


n_samples_per_class = 500
X, Y = make_samples(n_samples_per_class=n_samples_per_class,
                    class0_x0_mean=10.0, class0_x0_std=2.0, class0_x1_mean=20.0, class0_x1_std=2.0,
                    class1_x0_mean=20.0, class1_x0_std=2.0, class1_x1_mean=10.0, class1_x1_std=2.0,
                    class2_x0_mean=30.0, class2_x0_std=2.0, class2_x1_mean=20.0, class2_x1_std=2.0)

plt.subplot(1, 4, 1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.title("Groundtruth Data")

y_pred_kmeans = KMeans(n_clusters=3, random_state=1).fit_predict(X)
plt.subplot(1, 4, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_kmeans)
plt.title("KMeans")

y_pred_meanshift = MeanShift().fit_predict(X)
plt.subplot(1, 4, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_meanshift)
plt.title("MeanShift")

y_pred_gaussianmixture = GaussianMixture(n_components=3).fit_predict(X)
plt.subplot(1, 4, 4)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_gaussianmixture)
plt.title("Gaussian Mixture")

plt.show()
plt.close()

# Incorrect number of clusters
plt.subplot(4, 4, 1)
plt.title("Incorrect Number of Cluster")
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.ylabel("Ground truth")

y_pred_kmeans = KMeans(n_clusters=2, random_state=1).fit_predict(X)
plt.subplot(4, 4, 5)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_kmeans)
plt.ylabel("Kmeans")

y_pred_meanshift = MeanShift().fit_predict(X)
plt.subplot(4, 4, 9)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_meanshift)
plt.ylabel("MeanShift")

y_pred_gaussianmixture = GaussianMixture(n_components=2).fit_predict(X)
plt.subplot(4, 4, 13)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_gaussianmixture)
plt.ylabel("Gaussian Mixture")

# Anisotropicly distributed data
transformation = [[0.6, -0.6], [-0.4, 0.9]]
X_aniso = np.dot(X, transformation)
plt.subplot(4, 4, 2)
plt.title("Anisotropicaly Distributed")
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=Y)

y_pred_kmeans = KMeans(n_clusters=3).fit_predict(X_aniso)
plt.subplot(4, 4, 6)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_kmeans)

y_pred_meanshift = MeanShift().fit_predict(X_aniso)
plt.subplot(4, 4, 10)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_meanshift)

y_pred_gaussianmixture = GaussianMixture(n_components=3).fit_predict(X_aniso)
plt.subplot(4, 4, 14)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_gaussianmixture)

# Different variance
X_varied, Y_varied = make_samples(n_samples_per_class=n_samples_per_class,
                    class0_x0_mean=10.0, class0_x0_std=0.5, class0_x1_mean=20.0, class0_x1_std=0.5,
                    class1_x0_mean=20.0, class1_x0_std=2.0, class1_x1_mean=10.0, class1_x1_std=2.0,
                    class2_x0_mean=30.0, class2_x0_std=5.0, class2_x1_mean=20.0, class2_x1_std=5.0)
plt.subplot(4, 4, 3)
plt.title("Unequal Variance")
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=Y_varied)

y_pred_kmeans = KMeans(n_clusters=3).fit_predict(X_varied)
plt.subplot(4, 4, 7)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_kmeans)

y_pred_meanshift = MeanShift().fit_predict(X_varied)
plt.subplot(4, 4, 11)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_meanshift)

y_pred_gaussianmixture = GaussianMixture(n_components=3).fit_predict(X_varied)
plt.subplot(4, 4, 15)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred_gaussianmixture)

# Unevenly sized blobs
X_filtered = np.vstack((X[Y == 0][:n_samples_per_class], X[Y == 1][:int(n_samples_per_class/10)], X[Y == 2][:int(n_samples_per_class/100)]))
Y_filtered = np.hstack((Y[:n_samples_per_class], Y[n_samples_per_class:n_samples_per_class+int(n_samples_per_class/10)], Y[2*n_samples_per_class:2*n_samples_per_class+int(n_samples_per_class/100)]))
plt.subplot(4, 4, 4)
plt.title("Unevenly Sized Blobs")
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=Y_filtered)

y_pred_kmeans = KMeans(n_clusters=3, random_state=1).fit_predict(X_filtered)
plt.subplot(4, 4, 8)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred_kmeans)

y_pred_meanshift = MeanShift().fit_predict(X_filtered)
plt.subplot(4, 4, 12)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred_meanshift)

y_pred_gaussianmixture = GaussianMixture(n_components=3).fit_predict(X_filtered)
plt.subplot(4, 4, 16)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred_gaussianmixture)

plt.show()
