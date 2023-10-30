# https://numpy.org/doc/stable/reference/routines.linalg.html
import numpy as np

A = np.array([[1, -2], [0, 1], [1, 0]])
print("A:{}".format(A))
B = np.array([[1, -2, 1], [0, 1, 3]])
print("B:{}".format(B))

AB = np.matmul(A, B)
print("A x B:{}".format(AB))
AB = A @ B
print("A x B:{}".format(AB))
BA = B @ A
print("B x A:{}".format(BA))

A = np.array([1, 2, 3])
print("A:{}".format(A))
print("A norm:{}".format(np.linalg.norm(A)))

B = np.array([4, 5, 6])
print("B:{}".format(B))

print("A B dot product:{}".format(np.dot(A, B)))
print("A B outer product:{}".format(np.outer(A, B)))
print("A B cross product:{}".format(np.cross(A, B)))

A = np.array([[1, 2, 3], [-1, 3, 1], [1, 3, 6]])
print("A:{}".format(A))
print("A trace:{}".format(np.trace(A)))

b = np.array([[1], [2], [3]])
print("A:{}".format(A))
print("b:{}".format(b))

Ainv = np.linalg.inv(A)
print("Ainv:{}".format(Ainv))
print("A x Ainv:{}".format(A @ Ainv))
print("Ainv x A:{}".format(Ainv @ A))
print("Ainv x b:{}".format(Ainv @ b))
x = Ainv @ b
print("x:{}".format(x))
print("A @ x:{}".format(A @ x))

A = np.array([[1, 2, 3], [-1, 3, 1], [1, 2, 3]])
print("A:{}".format(A))
try:
    Ainv = np.linalg.inv(A)
except np.linalg.LinAlgError as ex:
    print(ex)
Apinv = np.linalg.pinv(A)
print("Apinv:{}".format(Apinv))
print("A x Apinv:{}".format(A @ Apinv))
print("Apinv x A:{}".format(Apinv @ A))
print("A rank:{}".format(np.linalg.matrix_rank(A)))
print("A determinant:{}".format(np.linalg.det(A)))

# Linear systems of equations
# x0 + 2 * x1 = 1
# 3 * x0 + 5 * x1 = 2
# A x = b
# A = [1 2
#      3 5]
# b = [1
#      2]
A = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution is", x)
