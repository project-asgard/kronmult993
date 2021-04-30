from functools import reduce
import numpy as np

def kronmult(matrix_list, x):
    """
    computes the product of the kronecker product of all the matrices in the list with X:
    kron(matrix_list) @ X
    does it by explicitely computing the kronecker product to minimize the likelyhood of error
    """
    matrix = reduce(np.kron, matrix_list)
    return matrix @ x
