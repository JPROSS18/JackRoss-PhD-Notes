****
A data-driven approach to finding the governing dynamical systems. SINDy works having a large library of non-linear candidate functions.

A snapshot of data is fitted to the library of candidate functions using spare-regression. Each function is fitted to the data using regularized regression with the constraint that most coefficients must be zero (sparsity). This assumes that most real-world systems have a small number of dominant terms. 

Once candidate functions have been selected, they can be used to deduce symbolic governing equations. 