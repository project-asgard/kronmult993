# Python's kronmult implementation

I am 90% such that this is a reinvention of [algorithm 993](Algorithm 993: Efficient Computation with Kronecker Products).

The comments in the code follow the notations from [ON KRONECKER PRODUCTS, TENSOR PRODUCTS AND MATRIX DIFFERENTIAL CALCULUS by Stephen Pollock](https://www.le.ac.uk/economics/research/RePEc/lec/leecon/dp14-02.pdf):

```cpp
R = (r_i e_i)     // a row vector
C = (c_j e^i)     // a column vector
A = (A_ij e_i^j)  // a matrix
B = (B_jk e_j^k)  // another matrix with a dimenssion in common
C = (C_kl e_k^l)  // another matrix with no dimenssion in common
A @ B = (A_ij e_i^j) @ (B_jk e_j^k) = ({A_ij B_jk} e_i^k)   // matrix product
kron(A,C) = (A_ij C_kl e_ik^jl)   // kronecker product
```

The colmajor and rowmajor storage orders can be translated as follows:

```cpp
colmajor(A_ij e_ij) = (A_ij e_j^i)
rowmajor(A_ij e_ij) = (A_ij e_j^i)
```

Our target, Asgard, uses colmajor.
