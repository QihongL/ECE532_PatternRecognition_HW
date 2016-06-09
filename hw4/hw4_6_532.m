%% Predict user ratings based on others
%% load the data matrix
clear all; clc; close all; 
load('jesterdata.mat')
[m,n] = size(X);
[U,S,V] = svd(X);

%% use the power method to approximate the first right singular vector
i = 0; 
v1 = randn(n,1);
while i < 50
    % perform the power iteration
    v1 = X'*X * v1 / norm(X'*X * v1,2);
    i = i+1;
    % record the error, stop when the error is small 
    error(i) = norm(abs(v1) - abs(V(:,1)));
    if error(i) < 1e-8
        break;
    end
end

%% get 1st L/R singular vectors and singular value
% now v1 is the 1st right singular vector
% sigma1 is the first singular value
sigma1 = norm(X * v1,2);
% u1 is the 1st left singular vector
u1 = X * v1 / sigma1;

%% visualize the performance
plot(error)
tt = sprintf('The L2 distance between the approximation and truth over iterations');
title(tt, 'fontsize', 14)
ylabel('error', 'fontsize', 14)
xlabel('iterations', 'fontsize', 14)