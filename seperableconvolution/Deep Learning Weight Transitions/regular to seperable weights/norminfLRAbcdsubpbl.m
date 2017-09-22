% function [v,countit] = norminfLRAbcdsubpbl(M,u) 
% 
% Algorithm to compute an optimal solution of the convex problem  
% 
%                min_v ||M - uv^T||_inf  
% 
% where ||M - uv^T||_inf = max_{i,j} |M_{ij} - u_i v_j|. 
%
% It uses a secant method described in the paper: 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 
%
% *** input ***
% M : an m-by-n matrix 
% u : an m-by-1 vector 
% 
% *** output *** 
% v       : the solution to min_v max_{j, i | u_i~=0} |M_{ij} - u_i v_j|
% countit : number of iterations of the secant method

function [v,countit] = norminfLRAbcdsubpbl(M,u) 

indunz = find( abs(u) > 1e-12 ); 
n = size(M,2); 

if isempty(indunz) % u = 0 => any v is solution, we select the mean value
    v = mean(M); 
else
    u = u(indunz); 
    su = sign(u); 
    u = u.*su; 
    m = length(u); 
    M = M(indunz,:).*repmat(su,1,n); 
    if length(indunz) == 1
        % Trivial solution
        v = M/u; 
    else
        % Secant method for norm-inf LRA 
        [~,i1] = min( M./repmat(u,1,n) ); 
        [~,i2] = max( M./repmat(u,1,n) ); 
        j = [0:n-1]; 
        v = ( M(i1 + j*m)' + M(i2 + j*m)' )./( u(i1) + u(i2) ); 
        if length(indunz) == 2
            return; 
        else
            vold = v+1; 
            countit = 0; 
            while norm(v-vold) > 1e-12 && countit <= 100 %safety procedure
                vold = v; 
                R = M-u*v'; 
                [~,ia] = max(abs(R)); 
                slope = -sign( R(ia + j*m  ) ); 
                indi1 = find(slope == -1); 
                i1(indi1) = ia(indi1); 
                indi2 = find(slope == 1); 
                i2(indi2) = ia(indi2);                 
                v = ( M(i1 + j*m)' + M(i2 + j*m)' )./( u(i1) + u(i2) );  
                countit = countit+1; 
            end
            % This should not happen (we have never observed it) 
            if countit == 101
                disp('Secant method requires more than 100 iterations...') 
                disp('Please report your instance to N. Gillis, thanks!'); 
            end
        end
    end
end