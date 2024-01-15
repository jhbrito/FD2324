import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics


file = 'winequality-white.csv'
dados = pd.read_csv(file)

print("INFO:{}".format(dados.info))
print("DESCRIBE:{}".format(dados.describe(include="all")))
print("DTYPES:{}".format(dados.dtypes))
print("Shape:{}".format(dados.shape))
print("HEAD:{}".format(dados.head()))
print("Columns:{}".format(dados.columns))

Y = np.array(dados["quality"])
dados.drop("quality", axis=1, inplace=True)
X = np.array(dados, dtype="float64")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, stratify=Y, random_state=1)

model = LinearSVC()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)


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
