% Example 2, testing the secant method, from the paper 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 
clear all; %clc; 
% Values of m 
m = [1e1 1e2 1e3 1e4]; % in the paper m = [1e1 1e2 1e3 1e4 1e5 1e6 1e7];
% number of columns of of M (n=1 => we generate them one by one)
n = 1; 
% number of problems solved 
itern = 1e2; % in the paper: 10^4 
for i = 1 : length(m) 
    for k = 1 : itern
        M = randn(m(i),n); 
        u = randn(m(i),1); 
        % Gaussian distribution 
        cput = cputime; 
        [v1,cnt1] = norminfLRAbcdsubpbl(M,u); 
        tG(i,k) = cputime-cput; 
        cntG(i,k) = cnt1; 
        % display evolution of computations 
        if (k - 30*floor(k/30)) == 0
            fprintf('%2.0f. \n', k); 
        else
            fprintf('%2.0f.', k); 
        end
    end
    fprintf('\n');  
    fprintf('Done with m=%2.2d \n', m(i));  
end
fprintf('\n'); 
% Display results
fprintf('--------------------------------------------------------------------------------------- \n'); 
fprintf('--------- Repartition of the number of iterations performed by the secant method ------ \n') 
fprintf('--------------------------------------------------------------------------------------- \n'); 
fprintf(' m / #it.   |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  Time(s). \n');
fprintf('--------------------------------------------------------------------------------------- \n'); 
for i = 1 : length(m)
    clear val; 
    for ct = 1 : 9
        val(ct) = sum( cntG(i,:) == ct ); 
    end
    fprintf('  %7.2d   | %3.2d  | %3.2d  | %3.2d  | %3.2d  | %3.2d  | %3.2d  | %3.2d  | %3.2d  | %3.2d  |  %1.2f \n', m(i), val, sum( tG(i,:)) );
end
fprintf('--------------------------------------------------------------------------------------- \n\n'); 