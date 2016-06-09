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

%% analysis 
% fit least square 
A = X(ratedJokesIdx,:);
y = b(ratedJokesIdx);
I = eye(numUsers);

% choose lambda, preallocate for errors
lambdas = 1e-1 :1e-1: 1;
prediction = nan(numJokes, length(lambdas));

% loop over all lambdas
for i = 1 : length(lambdas)
    lambda = lambdas(i);
    weights = inv(A'* A + lambda * I) * A' * y;
    % compute the prediction 
    prediction(:,i) = X * weights;
end
errors = bsxfun(@minus, prediction, trueb);

%% plot the test error 
% plot the test error 
plot(errors(~ratedJokesIdx,:))
title('Error for the unrated jokes', 'fontsize', 14)
xlabel('Unrated 75 Jokes', 'fontsize', 14)
% compute the mean absolute errors
mean(abs(errors(~ratedJokesIdx,:)))
% find the predicted best joke
for lambdaIdx = 1:10; 
    find(prediction(:,lambdaIdx) == max(prediction(:,lambdaIdx)));
end

