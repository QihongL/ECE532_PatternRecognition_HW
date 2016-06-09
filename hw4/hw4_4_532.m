%% Predict user ratings based on others
%% load the data matrix
clear all; clc; close all; 
% Each row is a joke, we have 100 jokes.  
% Each column is a user. We have 7200 users 
% Each of the users rated the quality of each joke on a scale of [?10, 10].
load('jesterdata.mat')
% load the partial data for the new user 
% b is the partial rating, the user rated x out of 100 jokes
load('newuser.mat')

%% read constant 
RATING.max = 10;
RATING.min = -10;
[numJokes, numUsers] = size(X);

% compute the singular value decomposition with the first 20 rows
[U,S,V] = svd(X, 'econ');

%% plot spec(X)
plot(diag(S));
title('Singular values from large to small')
rank(X)