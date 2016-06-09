% initialization
clear all; close all; clc;
load('BreastCancer.mat')
[m] = length(y);
% set some parameters
numLambdas = 20;
LAMBDAS = flip(logspace(-5,5,numLambdas));
trainSize = 100;
trainIdx = false(m,1);
trainIdx(1:trainSize) = 1;

% subset the data
Xtrain = X(trainIdx,:);
ytrain = y(trainIdx);
Xtest = X(~trainIdx,:);
ytest = y(~trainIdx);

% set parameter, lambda, learning rate, ...
tau = .9/ norm(X,2)^2;

%% compute the trade off
for i = 1 : numLambdas
    fprintf('%d\n',i);
    %% fit a lasso model for a particular lambda
    [beta, record] = lasso_lsta(Xtrain, ytrain, LAMBDAS(i), tau, false);
    
    %% compute the performance
    % performance on the training data
    prediction = Xtest * beta(:,end);
    residual(i) = norm(prediction - ytest,2);
    accuracy(i) = sum(sign(prediction) == ytest) / (m - trainSize);
    % beta norm 
    norm_beta(i) = norm(beta(:,end),1);
    % nonzero betas
    nnz(i) = record.nonZeroBetas;
end
%% plot the performance 
% plot the residual-norm trade off 
subplot(1,2,1)
plot(norm_beta, residual, 'linewidth', 1.5)
FZ = 14;
title('The trade off curve','fontsize', FZ)
xlabel('Norm: $\|\beta\|_1$','Interpreter','LaTex', 'fontsize', FZ)
ylabel('Residual: $\|X \beta - y\|_2$','Interpreter','LaTex', 'fontsize', FZ)
% plot the error-sparsity trade off 
subplot(1,2,2)
plot(nnz, 1-accuracy, 'linewidth', 1.5)
title('Error against the sparsity','fontsize', FZ)
xlabel('Number of non zero beta','fontsize', FZ)
ylabel('Error rate on the test set','fontsize', FZ)

