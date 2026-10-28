import numpy as np
y = np.array([1, 2, 3])
print(y)

# vector norm
def norm(x):
    return np.sqrt(sum(y**2 for y in x))
# let's test the norm function for the vector y
# first compute the norm by using the custom norm function
n = norm(y)
print("Norm of y using custom function:", n)

# now compute the norm using numpy's built-in function
n_np = np.linalg.norm(y)
print("Norm of y using numpy:", n_np)

# check if both norms are equal
if np.isclose(n, n_np):
    print("Both norms are equal.")
else:
    print("Norms are not equal.")

# check by an example 
norm_function = norm(y)
numpy_norm = np.linalg.norm(y)
if np.isclose(norm_function, numpy_norm):
    print("The custom norm function and numpy's norm function give the same result.")

# matrix-vector product
def matrix_vector_product(A, x):
    if len(x) != A.shape[1]:
        raise ValueError("Dimension mismatch: The length of vector x must match the number of columns in matrix A.")
    v = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v[i] += A[i, j] * x[j]
    return v

print("Matrix-vector product of A and y:", matrix_vector_product(np.array([[1, 2], [3, 4]]), y[:2]) )

# check by an example
A = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
result = matrix_vector_product(A, x)
print("Matrix-vector product of A and x:", result)
# also check with numpy's built-in function
result_np = A @ x
print("Matrix-vector product using numpy:", result_np)
if np.allclose(result, result_np):
    print("Both matrix-vector products are equal.")
    