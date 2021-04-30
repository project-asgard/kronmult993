import numpy as np
from kronmult import naive, rowmajor, colmajor

# dimensions of the matrices
n = 2
nvec = 3

# matrix to kron together
A1 = np.random.randint(0, 10, (n,n))
A2 = np.random.randint(0, 10, (n,n))
A3 = np.random.randint(0, 10, (n,n))
matrices = [A1, A2, A3]

# right hand vector
X = np.random.randint(0, 10, (n**3, nvec))

#------------------------------------------------------------------------------
# TEST RESHAPING OPERATIONS

def test_rowmajor_reinterpret():
    print("X\n", X)
    X_mat = rowmajor.reshape_nb_columns(X, nb_cols=n)
    print("Xmat\n", X_mat)
    X_flat = rowmajor.flatten(X_mat)
    print("Xflat\n", X_flat)
#test_rowmajor_reinterpret()

def test_colmajor_reinterpret():
    print("X\n", X)
    X_mat = colmajor.reshape_nb_columns(X, nb_cols=n)
    print("Xmat\n", X_mat)
    X_flat = rowmajor.flatten(X_mat)
    print("Xflat\n", X_flat)
#test_colmajor_reinterpret()

#------------------------------------------------------------------------------
# TEST VARIOUS KRONMULT IMPLEM

k_naive = naive.kronmult(matrices, X)
print("naive\n", k_naive)

k_row = rowmajor.kronmult(matrices, X)
print("row\n", k_row)

k_col = colmajor.kronmult(matrices, X)
print("col\n", k_col)
