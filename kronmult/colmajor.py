import numpy as np

#----------------------------------------------------------------------------------------
# REINTERPRET

def reshape_nb_rows(X, nb_rows):
    """
    reinterprets the vector X as a col major matrix with the given number of columns
    reshape_nb_columns( (A_ij e_ij) , j) = (A_ij e_j^i)
    this is a no-op in colmajor encoding
    """
    nb_cols = int(X.size / nb_rows)
    return np.reshape(X, (nb_rows,nb_cols), order='F')

def flatten(X):
    """
    reinterprets a col major matrix back to a flat vector
    flatten( (A_ij e_i^j) ) = (A_ij e_ji)
    this is a no-op in colmajor encoding
    """
    return np.reshape(X, X.shape, order='F')

#----------------------------------------------------------------------------------------
# VECTOR

def partial_kronmult(matrix, X):
    """
    given A = (A_ij e_i^j)
    and X = (X_kj e_kj)
    partial_kronmult(A, X) = ({A_ij X_kj} e_ik)
                           = flatten( (X_kj e_k^j) @ (A_ij e_j^i) )
                           = flatten( (X_kj e_j^k)^T @ A^T )
                           = flatten( reshape_nb_columns(X)^T @ A^T )

    this operation satisfies:
    kron(A1...AK) @ X = kron(A1...Ak-1) @ partial_kronmult_back(Ak,X)
    meaning that it can be chained to consume a kronmult one matrix at a time

    NOTE:
    - the X^T transposition can be included in the multiplication efficiently with a BLAS
    - A^T could be included in the multiplication (but would be bad for access pattern) or transposed beforehand
    - the reshape transformation are no-op if the data was already stored in colmajor
    - we could also do things like kron(A1...Ak) @ X = kron(A1...kron(AK-1, AK)) @ X to see how it impacts performances
    """
    _nb_rows, nb_cols = matrix.shape
    result = reshape_nb_rows(X, nb_rows=nb_cols).T @ matrix.T
    return flatten(result)

def kronmult_vector(matrix_list, X):
    """
    computes the product of the kronecker product of all the matrices in the list with X:
    kron(matrix_list) @ X
    does so by using an operation that consume the matrices *from the back of the list* to the front of the list one after the other

    NOTE:
    - this algorithm is only correct if X is a flat vector (not columns)
    - the matrices do not need to be square or of similar size as long as X has the correct size
    """
    result = X
    for matrix in reversed(matrix_list):
        result = partial_kronmult(matrix, result)
    return result

#----------------------------------------------------------------------------------------
# MATRIX

def kronmult(matrix_list, M):
    """
    same as kronmult_vector but designed to be used on a matrix
    rather than a flat vector
    """
    # stashes a columns in the proper place so that they will be taken into account during the computation
    X = flatten(M)
    # performs the kronmult
    result = kronmult_vector(matrix_list, X)
    # reinterpret the result as a matrix rather than a flat vector and puts the columns back in order
    return reshape_nb_rows(result, nb_rows=M.shape[1]).T
