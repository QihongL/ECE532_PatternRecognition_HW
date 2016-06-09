clear; close all;clc
training_data
% set lambda 
lambda = 1e-5;
% fit ridge regression vs. the dual optimization 
x.ridge = inv(A' * A + lambda * eye(size(A' * A))) * A' * b;
alpha   = inv(A * A' + lambda * eye(size(A * A'))) * b;
% recover the ridge weights from the dual
x.ridgeDual = A' * alpha;

[x.ridge x.ridgeDual]