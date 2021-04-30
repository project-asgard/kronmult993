import numpy as np

#----------------------------------------------------------------------------------------
# REINTERPRET

def reshape_nb_columns(X, nb_cols):
    """
    reinterprets the vector X as a col major matrix with the given number of columns
    reshape_nb_columns( (A_ij e_ij) , j) = (A_ij e_j^i)
    this is a no-op in colmajor encoding
    """
    nb_rows = int(X.size / nb_cols)
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
    and X = (X_jk e_jk)
    partial_kronmult(A, X) = ({A_ij X_jk} e_ki)
                           = flatten( ({A_ij X_jk} e_i^k) )
                           = flatten( (A_ij e_i^j) (X_jk e_j^k) )
                           = flatten( A rowmajor( (X_jk e_jk) ) )
                           = flatten( A rowmajor(X) )
                           = flatten( A reshape_nb_columns(X)^T )

    this operation satisfies:
    kron(A1...AK) @ X = reduce(X,A -> partial_kronmult(A,X) , A1..AK)
    meaning that it can be chained to consume a kronmult one matrix at a time

    NOTE:
    - A@B^T is *NOT* very efficient in colmajor, we should search for a A^T@B form instead
    - the colmajor transformation are no-op if the data was already stored in colmajor anyway
    - we could also do things like kron(A1...Ak) @ X = kron(A1...kron(AK-1, AK)) @ X to see how it impacts performances
    """
    _nb_rows, nb_cols = matrix.shape
    result = matrix @ reshape_nb_columns(X, nb_cols=nb_cols).T
    return flatten(result)

def kronmult_vector(matrix_list, X):
    """
    computes the product of the kronecker product of all the matrices in the list with X:
    kron(matrix_list) @ X
    does so by using an operation that consume the matrices from the front of the list one after the other

    NOTE:
    - this algorithm is only correct if X is a flat vector (not columns)
    - the matrices do not need to be square or of similar size as long as X has the correct size
    """
    result = X
    for matrix in matrix_list:
        result = partial_kronmult(matrix, result)
    return result

#----------------------------------------------------------------------------------------
# MATRIX

def flatten_matrix(A):
    """
    flattens a matrix in a colmajor way
    flatten_matrix( (A_ij e_i^j) ) = (A_ij e_ij)
    """
    return flatten(A.T)

def kronmult(matrix_list, M):
    """
    same as kronmult_vector but designed to be used on a matrix
    rather than a flat vector
    """
    # stashes a columns in the proper place so that they will be taken into account during the computation
    X = flatten_matrix(M)
    # performs the kronmult
    result = kronmult_vector(matrix_list, X)
    # reinterpret the result as a matrix rather than a flat vector
    return reshape_nb_columns(result, nb_cols=M.shape[1])
