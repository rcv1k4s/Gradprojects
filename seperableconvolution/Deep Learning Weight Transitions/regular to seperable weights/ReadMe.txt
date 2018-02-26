This code contains the following two main functions: 

1) norminfLRAbcd.m: A coordinate descent method (BCD) to tackle the l_infinity LRA problem

min_{U is m-by-r, V is r-by-n}  max_ij |M-UV|_ij . 

It can be tested on a simple example running "example4.m". It is Algorithm 1 described in the paper referenced below. 

2) DlinfR1A.m: Given M (m-by-n), k and a sign pattern s of dimension m, this function checks whether there exist (u,v) such that 
    (i) |Mij - ui vj| <= k for all i,j, 
    (ii) sign(u) = s.
It can be tested on a simple example running "example2.m". This problem can be solved via linear programming (see Lemma 1 in the paper below) and we use CVX (http://cvxr.com/) to do so (CVX needs to be installed to use this function). 


You can also run the other experiments from the paper below using example1.m, example3.m and experimentquantized.m. 


Reference: N. Gillis and Y. Shitov, Low-Rank Matrix Approximation in the Infinity Norm, 2017. 
