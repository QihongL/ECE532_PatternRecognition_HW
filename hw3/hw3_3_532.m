%% hw3-1-ece532
clear all; clc;
%% a) apply my gram-schimit to problem 1 
A = [3 0 0; 1 3 4]';
b = [1 3 1]';
% compute weights using the normal equation
beta_1 = A * inv(A' * A) * A' * b;
% compute the weights using my gramSchmidit 
Q = gramSchmidtProcess(A);
beta_2 = Q * Q' * b;
% One can verify that beta1 and beta2 are the same 


%% b) Test my Gram-schimidt with the following matrices A1,A2
A1 = [3 0 0 6; 1 3 4 1; 2 3 4 4]';
A2 = [1 0 0 3; 1 3 4 1; 2 3 4 4]';
% orthogonalize them
Q1 = gramSchmidtProcess(A1)
Q2 = gramSchmidtProcess(A2)
% check ranks 
if (rank(A1) == size(Q1,2) && rank(A2) == size(Q2,2))
    disp('The ranks are correct');
end
