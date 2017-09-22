% Example 4 on a small quantized low-rank matrix from the paper 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 

clc; 

m = 8; 
n = 5; 
r = 3; 

Utrue = randn(m,r); 
Vtrue = randn(r,n); 
Mtrue = Utrue*Vtrue 

Mquant = round(Mtrue)

[U,V,e,t] = norminfLRAbcd(Mquant,3); 

disp('The approximation U*V of Mquant is given by') 
U*V 
fprintf('with error %2.2f. \n', e(end)); 
fprintf('The error of L2 LRA is %2.2f. \n', e(1)); 