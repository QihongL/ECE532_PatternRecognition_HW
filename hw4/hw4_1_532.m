%% Predict user ratings based on others
%% load the data matrix
clear all; clc; close all; 
% Each row is a joke, we have 100 jokes.  
% Each column is a user. We have 7200 users 
% Each of the users rated the quality of each joke on a scale of [-10, 10].
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
% select the first 20 columns 
Xsubset = X(:,1:20);

%% analysis 
% fit least square 
A = Xsubset(ratedJokesIdx,:);
y = b(ratedJokesIdx);
weights = inv(A'* A) * A' * y;

% compare to the truth 
prediction = Xsubset * weights;
testerror = prediction - trueb;

% plot the test error 
subplot(1,2,1)
plot(testerror(ratedJokesIdx))
ylim([min(testerror)-3 max(testerror)+3])
title('Error for the rated jokes', 'fontsize', 14)
ylabel('difference between prediction and turth', 'fontsize', 14)
xlabel('Rated 25 Jokes', 'fontsize', 14)
subplot(1,2,2)
plot(testerror(~ratedJokesIdx))
ylim([min(testerror)-3 max(testerror)+3])
title('Error for the unrated jokes', 'fontsize', 14)
xlabel('Unrated 75 Jokes', 'fontsize', 14)
% mean test error 
mean(abs(testerror(ratedJokesIdx)))
mean(abs(testerror(~ratedJokesIdx)))
% predict favoriate joke
find(prediction == max(prediction))