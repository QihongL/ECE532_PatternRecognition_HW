% read data
clear all; close all; clc;
load('BreastCancer.mat')

% set parameter, lambda, learning rate, ...
tau = .9/ norm(X,2)^2;
lambda = .1;

% fit lasso
[finalbeta.ista] = lasso_lsta(X, y, lambda, tau, 0);


[finalbeta.gd] = lasso_gd(X,y, lambda, tau)