import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

####
# Comparison of LDA and PCA 2D projection of Iris dataset
# The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour and Virginica) with 4 attributes: sepal length, sepal width, petal length and petal width.
#
# Principal Component Analysis (PCA) applied to this data identifies the combination of attributes (principal components, or directions in the feature space) that account for the most variance in the data. Here we plot the different samples on the 2 first principal components.
#
# Linear Discriminant Analysis (LDA) tries to identify attributes that account for the most variance between classes. In particular, LDA, in contrast to PCA, is a supervised method, using known class labels.
#
###

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])


ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=y)
ax.set_xlabel(feature_names[1])
ax.set_ylabel(feature_names[2])
ax.set_zlabel(feature_names[3])

ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=y)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[2])
ax.set_zlabel(feature_names[3])

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print("PCA explained variance ratio (first two components): {}".format(str(pca.explained_variance_ratio_)))
ax = fig.add_subplot(2, 3, 4)
ax.scatter(X_r[:, 0], X_r[:, 1], c=y, alpha=0.8)
ax.set_title("PCA of IRIS dataset")

kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit(X).transform(X)
ax = fig.add_subplot(2, 3, 5)
ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, alpha=0.8)
ax.set_title("kPCA of IRIS dataset")

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
# Percentage of variance explained for each components
print("LDA explained variance ratio (first two components): {}".format(str(lda.explained_variance_ratio_)))
ax = fig.add_subplot(2, 3, 6)
ax.scatter(X_r2[:, 0], X_r2[:, 1], c=y, alpha=0.8)
ax.set_title("LDA of IRIS dataset")

plt.show()
