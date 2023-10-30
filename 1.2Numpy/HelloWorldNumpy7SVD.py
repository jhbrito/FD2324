# SVD -> A = U * S * Vt
import numpy as np
import os
import cv2 as cv2

verbose = False
folder = "../Files"

A = cv2.imread(os.path.join(folder, "dog.jpg"), 0)
A = A / 255.0

mr = np.linalg.matrix_rank(A)
print("matrix rank:{}".format(mr))

U, S, Vh = np.linalg.svd(A)
if verbose:
    print("U", U)
    print("S", S)
    print("Vh", Vh)

# all singular values
Sd = np.zeros(A.shape)
if verbose:
    print("Sd", Sd)
n_percentage = 1.0
n = int(np.round(len(S)*n_percentage))
print("Using {} singular values ({}%)".format(n, n_percentage*100))
for i in range(n):
    Sd[i, i] = S[i]
if verbose:
    print("Sd", Sd)

# Ar = np.matmul(np.matmul(U, Sd), Vh)
Ar = U @ Sd @ Vh
if verbose:
    print("Ar", Ar)

dif = np.sum(np.abs(A-Ar))
print("dif: {}".format(dif))

# 10% of singular values

Sd1 = np.zeros(A.shape)
if verbose:
    print("Sd", Sd1)
n_percentage = 0.1
n = int(np.round(len(S) * n_percentage))
print("Using {} singular values ({}%)".format(n, n_percentage * 100))
for i in range(n):
    Sd1[i, i] = S[i]
if verbose:
    print("Sd1", Sd1)

# Ar = np.matmul(np.matmul(U, Sd), Vh)
Ar1 = U @ Sd1 @ Vh
if verbose:
    print("Ar1", Ar1)

dif1 = np.sum(np.abs(A - Ar1))
print("dif 10%: {}".format(dif1))
print("dif per pixel 10%: {}".format(255.0*dif1/(A.shape[0]*A.shape[1])))


cv2.imshow("A", A)
cv2.imshow("Ar", Ar)
cv2.imshow("Ar 10%", Ar1)
cv2.waitKey(0)
