function [finalAcc] = blurData()
%% load data
load('blurringData.mat')

%% set parameters
[m,n] = size(X);
CVBSize = 53;
maxRank = n;
LAMBDAS = [0 2.^(-1 : 4)];  % L2 reg. parameters that we want to try
I = eye(n);

%% set up the final hold-out set
[finalHoldoutIdx, K_full] = setupCVBlocks(size(X,1), CVBSize);

% loop over all possible final hold out set (8 of them)
for k = 1 : K_full;
    % hold out 1 chunk as the final test set
    X_part = X(~finalHoldoutIdx(:,k),:);
    y_part = y(~finalHoldoutIdx(:,k));
    X_final = X(finalHoldoutIdx(:,k),:);
    y_final = y(finalHoldoutIdx(:,k));
    
    %% setup tuning sets cross-validation blocks for the tunning procedure
    [holdoutIdx, K_tune] = setupCVBlocks(size(X_part,1), CVBSize);
    % loop over all possible tuning hold out set (7 of them)
    for i = 1:K_tune
        % split the data into training and test set
        X_train = X_part(~holdoutIdx(:,i),:);
        y_train = y_part(~holdoutIdx(:,i));
        X_test = X_part(holdoutIdx(:,i),:);
        y_test = y_part(holdoutIdx(:,i));
        
        testAccuracy = nan(maxRank,1);
        % compute the singular value decomposition
        [U,S,V] = svd(X_train, 'econ');
        for rank = 1:maxRank;
            %% fit the model
            S_inv_truncate = truncateS(inv(S), rank);
            % compute the beta with U,S,V
            beta(:,rank) = V * S_inv_truncate * U' * y_train;
            % make the prediction
            predict = ones(CVBSize,1);
            predict(X_test * beta(:,rank) <= 0) = -1;
            % evaluate the performance
            correctPred = bsxfun(@eq, predict, y_test);
            testAccuracy(rank) = sum(correctPred) / CVBSize;
        end
        
        %% fit the final model with the best rank
        % find the best rank
        bestRank = find(testAccuracy == max(testAccuracy) ,1, 'last');
        % compute the singular value decomposition
        [U,S,V] = svd(X_part, 'econ');
        S_inv_truncate = truncateS(inv(S), bestRank);
        % compute the beta with U,S,V
        beta = V * S_inv_truncate * U' * y_part;
        % make the prediction
        predict = ones(CVBSize,1);
        predict(X_final * beta <= 0) = -1;
        % evaluate the performance
        correctPred = bsxfun(@eq, predict, y_final);
        finalAcc.svd(i,k) = sum(correctPred) / CVBSize;
        
        %% fitting ridge regression 
        testAccuracy = nan(length(LAMBDAS),1);
        % compute SVD
        [U,S,V] = svd(X_train, 'econ');
        for l = 1:length(LAMBDAS)
            % choose a lambda value 
            lambda = LAMBDAS(l);
            %% fit the model
            % compute the beta with U,S,V
            beta(:,l) = V * inv(S^2 + I*lambda) * S * U' * y_train;
            % compute the beta with the normal equations
            beta_test = inv(X_train' * X_train + I*lambda) * X_train' * y_train;
            % check if they are the same
            if any(abs(beta_test - beta(:,l)) > 1e-12)
                warning('WARNING: SVD solution and normal equations solution are different!')
            end
            % make the prediction
            predict = ones(CVBSize,1);
            predict(X_test * beta(:,l) <= 0) = -1;
            % evaluate the performance
            correctPred = bsxfun(@eq, predict, y_test);
            testAccuracy(l) = sum(correctPred) / CVBSize;
        end
        
        %% fit the final model
        % find the best rank
        bestLambdaIdx = find(testAccuracy == max(testAccuracy) ,1, 'first');
        finallambda = LAMBDAS(bestLambdaIdx);
        % compute SVD
        [U,S,V] = svd(X_part, 'econ');
        % compute the beta with U,S,V
        beta = V * inv(S^2 + I*finallambda) * S * U' * y_part;
        % make the prediction
        predict = ones(CVBSize,1);
        predict(X_final * beta <= 0) = -1;
        %% evaluate the performance
        correctPred = bsxfun(@eq, predict, y_final);
        finalAcc.ridge(i,k) = sum(correctPred) / CVBSize;
        
    end % end of one tuning set 
end % end of one final hold out set

%% compare performance
mean(finalAcc.svd(:))
mean(finalAcc.ridge(:))

error.svd = 1 - finalAcc.svd;
error.ridge = 1 - finalAcc.ridge;
[H,P,CI,STATS] = ttest(error.svd(:),error.ridge(:))

end % end of definition of the main 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set up the cv blocks
function [holdOutIdx, numFolds] = setupCVBlocks(numData, cvBlockSize)
numFolds = numData/cvBlockSize;
holdOutIdx = false(numData,numFolds);        % the indicies for the hold out set
% get the hold out set indices for K folds
for i = 1 : numFolds
    holdOutIdx(((i-1)*cvBlockSize+1):(i)*cvBlockSize,i) = true;
end
end

%% truncate the diagnal matrix S: leave the first r diagnal entries
function S_truncate = truncateS(S, rank)
% truncate singular values
S_truncate = S;
for i = rank+1 : length(diag(S))
    S_truncate(i,i) = 0;
end
end
