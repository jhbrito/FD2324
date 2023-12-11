# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

np.random.seed(1)

n_samples_per_class = 100

class0_x0_mean = 10.0
class0_x0_std = 2.0
class0_x1_mean = 20.0
class0_x1_std = 2.0

class1_x0_mean = 20.0
class1_x0_std = 2.0
class1_x1_mean = 10.0
class1_x1_std = 2.0

X = np.zeros((2 * n_samples_per_class, 2), dtype=float)
Y = np.zeros((2 * n_samples_per_class), dtype=int)

X[0:n_samples_per_class, 0] = np.random.randn(n_samples_per_class) * class0_x0_std + class0_x0_mean
X[0:n_samples_per_class, 1] = np.random.randn(n_samples_per_class) * class0_x1_std + class0_x1_mean
X[n_samples_per_class:, 0] = np.random.randn(n_samples_per_class) * class1_x0_std + class1_x0_mean
X[n_samples_per_class:, 1] = np.random.randn(n_samples_per_class) * class1_x1_std + class1_x1_mean

Y[n_samples_per_class:2 * n_samples_per_class] = np.ones((n_samples_per_class), dtype=float)

fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 1)
ax.scatter(X[Y==0, 0], X[Y==0, 1], c='r', marker='o')
ax.scatter(X[Y==1, 0], X[Y==1, 1], c='g', marker='o')
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_title("Data")
# plt.show()
# plt.close(fig)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 2)

ax.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1], c='r', marker='o')
ax.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1], c='g', marker='o')

ax.scatter(X_test[Y_test==0, 0], X_test[Y_test==0, 1], c='m', marker='x')
ax.scatter(X_test[Y_test==1, 0], X_test[Y_test==1, 1], c='y', marker='x')

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_title("Split Data")
# plt.show()
# plt.close(fig)

model = LinearSVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = AdaBoostClassifier()
# model = MLPClassifier()

model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

ax = fig.add_subplot(1, 3, 3)

ax.scatter(X_test[Y_predict==0, 0], X_test[Y_predict==0, 1], c='r', marker='*')
ax.scatter(X_test[Y_predict==1, 0], X_test[Y_predict==1, 1], c='g', marker='*')

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_title("Predicted Classes")

plt.show()
plt.close(fig)


cm = metrics.confusion_matrix(Y_test, Y_predict, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test, Y_predict)
print("Precision Recall F-score Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test, Y_predict)
print("Classification Report:")
print(cr)
