% function [U,V,e,t] = norminfLRAbcd(M,r,U0,V0,maxiter,timelimit,relerr) 
%
% Block coordinate descent (BCD) method to tackle the following optimization
% problem 
% 
%                min_{U,V} ||M - UV^T||_inf  
% 
% where ||M - UV||_inf = max_{i,j} |M-UV|_{i,j}. 
%
% It optimizes alternatively over the columns of U and rows of V; 
% see the paper: 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 
%
% See example4.m for a simple example to use norminfLRAbcd 
%
% *** Input ***
% M       : an m-by-n matrix 
% r       : the factorization rank
%        - default: r=1
% U0, V0  : an m-by-r matrix and an r-by-n matrix
%        - default: best rank-r approximation in the Frobenius norm (SVD)
% maxiter : maximum number of iterations of the BCD method 
%        - default: 500
% timelim : maximum time in seconds alloted to the algorithm 
%        - default: 20 seconds
% relerr  : algorithm stopped if relative error between two iterations < relerr
%        - default: 1e-6 
% 
% *** Output *** 
% (U,V)   : the point obtained by the BCD method
% (e,t)   : error and time throughout iterations; use plot(t,e) to see the
%           evolution of the error over time (and plot(e) over iterations). 

function [U,V,e,t] = norminfLRAbcd(M,r,U0,V0,maxiter,timelimit,relerr) 

initT = cputime; 
mM = norminfty(M); % will be used for the relative error 
if nargin < 2
    r = 1; 
end
if nargin < 4 || isempty(U0) || isempty(V0) 
    [Usvd,Ssvd,Vsvd] = svds(M,r); 
    U0 = Usvd*Ssvd; 
    V0 = Vsvd'; 
end
if nargin < 5 || isempty(maxiter) 
    maxiter = 500; 
end
if nargin < 6 || isempty(timelimit) 
    timelimit = 20;
end
if nargin < 7 || isempty(relerr)  
    relerr = 1e-6;
end
iter = 1; 
e(iter) = norminfty(M-U0*V0); 
t(iter) = cputime-initT; 
U = U0; 
V = V0; 
while iter <= maxiter && cputime-initT <= timelimit 
    R = M-U*V; 
    for p = 1 : r
        % Update U(:,p) and V(p,:)
        R = R+U(:,p)*V(p,:); 
        U(:,p) = norminfLRAbcdsubpbl(R',V(p,:)'); 
        % For the NMF variant: use U(:,p) = max(0,U(:,p)); 
        V(p,:) = norminfLRAbcdsubpbl(R,U(:,p))'; 
        % For the NMF variant: use V(p,:) = max(0,V(p,:)); 
        R = R-U(:,p)*V(p,:); 
    end  
    iter = iter + 1; 
    e(iter) = norminfty(R); 
    t(iter) = cputime-initT;
    % Algorithm is stopped if the difference in relative error < relerr
    if e(iter-1)-e(iter) < relerr*mM
        fprintf('The BCD algorithm has converged in %2.0f iterations. \n', iter); 
        return;
    end
end