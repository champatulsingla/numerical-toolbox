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

