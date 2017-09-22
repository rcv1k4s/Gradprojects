% function norminfA = norminfty(A) 
%
% This function compute component-wise infinity norm of matrix A: 
%
% norminfA = max_{i,j} |A_{ij}| 

function norminfA = norminfty(A) 

norminfA = max( abs(A(:)) ); 