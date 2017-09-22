% Example 2 from the paper 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 
clear all; clc; 

disp('For the matrix') 
M1 = [  2   0   1   1  -1    
-1   2  -1  -1   0     
-1   1   2  -1  -1    
-1   1   1   2  -1    
1  -1   0   1   2 ] 
[u1,v1,e1,t1] = norminfLRAbcd(M1,1); 
disp('There exists a solution with error < 3/2, given by') 
u1*v1
fprintf('with error %2.4f. \n', e1(end));

fprintf('We can check if there exists a better solution with the same \n') 
fprintf('sign pattern for u using D-linf-R1A, using CVX (http://cvxr.com/): \n') 
[ucheck,vcheck,answer] = DlinfR1A(M1,e1(end)-0.0001,sign(u1)); 
fprintf(['The answer to D-linf-R1A with k=%2.4f is ' answer '.\n'],e1(end)-0.0001); 

disp('-----------------------------------------------------'); 

disp('For the matrix')
 M2 = [2     1    1    -1     1   
    -1     2    -1    -1     0    
    -1     0     2     1     1    
     0     1    -1     2    -1    
    -1    -1    -1     1     2] 
disp('There does not exist a solution with error < 3/2.')
[u2,v2,e2,t2] = norminfLRAbcd(M2,1); 
fprintf('The error is %2.2f. \n', e2(end));