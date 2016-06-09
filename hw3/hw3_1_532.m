%% hw3-1-ece532
clear all; 
A = [3 0 0; 1 3 4]';
b = [1 3 1]';
% Q is an Orthogonal matrix that I came up for A 
Q = [1 0 0; 0 3/5 4/5]';

% compute beta in two ways
beta_1 = A * inv(A' * A) * A' * b;
beta_2 = Q * Q' * b;

% check if they are the same
all(beta_1 == beta_2)