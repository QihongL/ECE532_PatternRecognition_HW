% initialization
clear all; close all; clc;
load('BreastCancer.mat')
rng(1)

% set some parameters
[m,n] = size(X);
numLambdas = 20;
LAMBDAS = flip(logspace(-4,4,numLambdas));
tau = .9/ norm(X,2)^2;
K = 10;

%% final hold out set
% generate CV indices block for the final hold out set
CVIDX = crossvalind('Kfold', m, K);
CVB = false(m,K);
for k = 1 : K
    CVB(CVIDX == k,k) = true;
end

%% hold out 1 chuck of the data as the final test set
% k from 1 to 10
LASSO_BEST = cell(K,1);
RIDGE_BEST = cell(K,1);
for k = 1:K
    finalHoldoutIdx = CVB(:,k);
    Xtune = X(~finalHoldoutIdx, :);
    ytune = y(~finalHoldoutIdx);
    Xfinal = X(finalHoldoutIdx, :);
    yfinal = y(finalHoldoutIdx);
    
    %% set up CV block for the training set (weight search)
    % generate CV indices block
    CVIDX_b = crossvalind('Kfold', m - length(yfinal), K-1);
    CVB_b = false(m - length(yfinal), K-1);
    for kk = 1 : K-1
        CVB_b(CVIDX_b == kk, kk) = true;
    end
    
    %% inner cross-validation
    for kk = 1 : K-1
        
        % split training vs. test data
        testSetIdx = CVB_b(:,kk);
        Xtrain = Xtune(~testSetIdx, :);
        ytrain = ytune(~testSetIdx);
        Xtest = Xtune(testSetIdx, :);
        ytest = ytune(testSetIdx);
        
        %% loop over all values of regularization parameter lambda
        % set learning rate
        for l = 1 : numel(LAMBDAS)
            fprintf('%d - %d - %d \n', k, kk, l);
            lambda = LAMBDAS(l);
            % fit lasso model
            [tempbeta, ~] = lasso_lsta(Xtrain, ytrain, lambda, tau, 0);
            lasso.beta(:,l) = tempbeta(:,end);
            lasso.lambda(l) = lambda;
            lasso.accuracy(l) = sum(sign(Xtest * lasso.beta(:,l)) == ytest) / length(ytest);
            % fit ridge model
            [tempbeta, ~] = ridge_iter(Xtrain, ytrain, lambda, tau, 0);
            ridge.beta(:,l) = tempbeta(:,end);
            ridge.lambda(l) = lambda;
            ridge.accuracy(l) = sum(sign(Xtest * ridge.beta(:,l)) == ytest) / length(ytest);
        end
        
        %% find the best beta and the associated parameter
        % lasso
        bestIdx = find(lasso.accuracy == max(lasso.accuracy),1);
        LASSO_BEST{k}.beta(:,kk) = lasso.beta(:,bestIdx);
        LASSO_BEST{k}.lambda(kk) = lasso.lambda(bestIdx);
        % ridge
        bestIdx = find(ridge.accuracy == max(ridge.accuracy),1);
        RIDGE_BEST{k}.beta(:,kk) = ridge.beta(:,bestIdx);
        RIDGE_BEST{k}.lambda(kk) = ridge.lambda(bestIdx);
        
    end
    
    %% compute the performance on the final hold out set 
    for kk = 1 : K-1
        prediction = Xfinal * LASSO_BEST{k}.beta(:,kk);
        LASSO_BEST{k}.diff(kk) = norm(prediction - yfinal,2); 
        LASSO_BEST{k}.accuracy(kk) =  sum(sign(prediction) == yfinal)/length(yfinal);
        
        prediction = Xfinal  * RIDGE_BEST{k}.beta(:,kk);
        RIDGE_BEST{k}.diff(kk) = norm(prediction - yfinal,2); 
        RIDGE_BEST{k}.accuracy(kk) =  sum(sign(prediction) == yfinal)/length(yfinal);
    end
end
