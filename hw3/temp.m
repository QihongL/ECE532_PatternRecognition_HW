clear all; clc; close all; 
m = 300; 
n = 2;
X = randn(m,n);
Y = sum(X,2) + randn(m,1);
% coeff = pca()

scatter3(X(:,1),X(:,2),Y)

pca(X)
