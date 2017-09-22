% Recovery of quantized low-rank matrices from Section 4 of the paper 
% Low-Rank Matrix Approximation in the Infinity Norm, Nicolas Gillis and 
% Yaroslav Shitov, 2017. 

clear all; clc; 
tic; 
% Tested dimensions
m = 100; % In the paper: m = 200; 
n = m; 
R = [1 2 5]; % In the paper:  R = [1 2 5 10 20];
% Number of generated matrices 
numtest = 20; % In the paper: numtest = 100 
% parameters 
maxiter = 1000; 
timelimit = Inf; 
precision = 1e-6; 
for gauss = 1 : 1  % Use 'gauss = 1 : 2' to test the uniform distribution
    for ir = 1 : length(R) 
        r = R(ir); 
        for i = 1 : numtest
            % Generate example 
            if gauss == 1
                Utrue = randn(m,r); 
                Vtrue = randn(r,n); 
            else
                Utrue = rand(m,r); 
                Vtrue = rand(r,n); 
            end
            Mtrue = Utrue*Vtrue; 
            Mquant = round(Mtrue); 
            % Compute l2 solution 
            ttsvd = cputime; 
            [u0,s0,v0] = svds(Mquant,r); 
            tsvd(ir,gauss,i) = cputime-ttsvd; 
            esvd(ir,gauss,i) = norminfty(Mquant - u0*s0*v0'); 
            % Compute l_inf LRA solution, initialized with l2 
            [Urecover,Vrecover,e,t] = norminfLRAbcd(Mquant,r,u0*s0,v0',maxiter,timelimit,precision);
            einf(ir,gauss,i) = e(end); 
            numit(ir,gauss,i) = length(e)-1; 
            timinf(ir,gauss,i) = t(end); 
        end
    end
end
% Display results  
for gauss = 1 : 1
    disp('--------------------------------------------------------------------------------------------------');
    if gauss == 1
        fprintf('     Recovery of quantized low-rank matrices - Case m=%3.0f, n=%3.0f, Gaussian distribution\n',m,n);
    else
        fprintf('     Recovery of quantized low-rank matrices - Case m=%3.0f, n=%3.0f, Uniform distribution\n',m,n);
    end
    disp('--------------------------------------------------------------------------------------------------');
    fprintf(' r   |      error of BCD     |  # runs with |   it. of BCD    |  Time (s.)  |     error of L2   \n')
    fprintf('     |  [ min , mean , max ] | error <= 0.5 | [min,mean,max]  |             | [ min , mean , max ]   \n')
    disp('--------------------------------------------------------------------------------------------------');
    for ir = 1 : length(R)
        fprintf('%2.0f   |   %1.2f , %1.2f , %1.2f  |   %3.0f/%3.0f    | %3.0f , %3.0f , %3.0f |    %2.2f     |  %1.2f , %1.2f , %1.2f \n' , ...
            R(ir), min(einf(ir,gauss,:)), mean(einf(ir,gauss,:)), max(einf(ir,gauss,:)), ...
            sum(einf(ir,gauss,:) <= 0.5), length(einf(ir,gauss,:)), ...
            min(numit(ir,gauss,:)), mean(numit(ir,gauss,:)), max(numit(ir,gauss,:)), ...
            mean(timinf(ir,gauss,:)), ...
            min(esvd(ir,gauss,:)), mean(esvd(ir,gauss,:)), max(esvd(ir,gauss,:) ));
    end
    disp('--------------------------------------------------------------------------------------------------');
end
toc; 