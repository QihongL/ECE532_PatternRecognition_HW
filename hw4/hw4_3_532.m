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

% check the information provided
ratedJokesIdx = (logical(b >= RATING.min) & logical(b <= RATING.max));
numRatedJokes = sum(ratedJokesIdx);

%% fit the model
A = X(ratedJokesIdx,:);
y = b(ratedJokesIdx);
I = eye(numUsers);
lambda = 1;
% fit ridge regression
weights = inv(A'* A + lambda * I) * A' * y;
% compute the prediction
prediction = X * weights;

%% find indices for the largest and second largest weights 
idx.largest1 = find(abs(weights) == max(abs(weights)));   % max 
idx.largest2 = find(abs(weights) == max(weights(abs(weights) < max(abs(weights)))));  % 2nd largest 
% get the data matrix 
Anew = [A(:,idx.largest1) A(:,idx.largest2)];
Xnew = [X(:,idx.largest1) X(:,idx.largest2)];
% figure out the weights
weightsNew = inv(Anew' * Anew) * Anew' * y;
% generate the prediction
prediction = Xnew * weightsNew;
% compute the error 
error = prediction - trueb;

%% show the error 
% plot the test error 
subplot(1,2,1)
plot(error(ratedJokesIdx))
title('Error for the rated jokes', 'fontsize', 14)
ylabel('difference between prediction and turth', 'fontsize', 14)
xlabel('Rated 25 Jokes', 'fontsize', 14)
subplot(1,2,2)
plot(error(~ratedJokesIdx))
title('Error for the unrated jokes', 'fontsize', 14)
xlabel('Unrated 75 Jokes', 'fontsize', 14)

% compute mean error
mean(error(ratedJokesIdx))
mean(error(~ratedJokesIdx))