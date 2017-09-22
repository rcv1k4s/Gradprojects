% function [u,v,answer] = DlinfR1A(M,k,su)  
%
% CVX code to solve D-l_inf-R1A when the sign of u is known, that is: 
% Given M and k, check whether there exists (u,v) such that 
% |Mij - ui vj| < k, 
% where u has the given sign pattern. 
% 
% You need to install CVX; see http://cvxr.com/cvx/ 
%
% 
% *** Input *** 
% M  : m-by-n matrix 
% k  : value 
% su : sign pattern of a solution u 
%     - default: = ones(m,1) (that is, u >= 0).  
% 
% *** Output *** 
% If there exists a solution to the instance (M,k) where u has sign 
% pattern su, it returns a solution (u,v) and answer = 'Yes'; 
% Otherwise it returns answer = 'No'.  
% 
% *** Example ***   
% ut = [1 2 3 4]'; vt = [1 2 3 4 5]; 
% M = ut*vt; 
% [u,v,answer] = DlinfR1A(M,0)
% fprintf('Error ||M-uv||_inf = %2.2d \n', norminfty(M-u*v)); 

function [u,v,answer] = DlinfR1A(M,k,su) 

if nargin <= 2
    su = ones( size(M,1) , 1);
end

% Keep only row and column indices of M with an entry > k 
rowi = find( max(abs(M)') > k ); 
columni = find( max(abs(M)) > k ); 

Ms = M(rowi, columni); 
su = su(rowi); 
[ms,ns] = size(Ms);
 
% flip sign according to su 
Ms = Ms.*(repmat(su,1,ns)); 

cvx_begin quiet 
    variable sus(ms,1)
    variable vs(ns,1);
    minimize(  1  );
    subject to
        for i = 1 : ms
            for j = 1 : ns
                sus(i)*(Ms(i,j)-k) <= vs(j); 
                vs(j) <= sus(i)*(Ms(i,j)+k); 
            end
        end
        sus(:) >= 1; 
cvx_end
 
[m,n] = size(Ms); 
s = zeros(m,1); 
s(rowi) = sus; 
v = zeros(1,n); 
v(columni) = vs; 
u = zeros(m,1); 
u(rowi) = (1./sus).*su; 
if sum(isnan(u)) >= 1 || min(sus) < 1 || cvx_status(1) ~= 'S' % Solved
    answer = 'NO'; 
else
    answer = 'YES';
end