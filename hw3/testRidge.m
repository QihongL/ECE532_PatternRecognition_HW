%% help me to think about ridge regression
%% initilization
clear all;clc;
rng(1)

% create data
m = 50;
n = 1;
X = randn(m,n);
Y = X + randn(m,n);

%% fit the models
% OLS
beta1 = inv(X' * X) * X' * Y;
% ridge
lambda = 0.8;
beta2 = inv(X' * X + lambda * eye(n,n)) * X' * Y;

%% plot
if true
    % show the data
    scatter(X,Y)
    hold on
    % plot the OLS regression line
    olsline = refline(beta1,0)
    % plot the ridge regression line
    olsline.Color = 'r';
    refline(beta2,0)
    
    legend({'data','OLS','RIDGE'})
end