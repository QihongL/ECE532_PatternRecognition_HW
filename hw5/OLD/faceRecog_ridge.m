function [] = faceRecog_ridge()
%% load data
load('face_emotion_data.mat')
%% initialization
addNewFeatures = 0;
if addNewFeatures
    % create 3 new features as random linear combinations of original features
    rng(1);
    newFeatures = X * randn(size(X,2),3);
    X = [X newFeatures];
end

%% set parameters
[m,n] = size(X);
CVBSize = 16;       % cross validation block size (diviside by m)
LAMBDAS = [0 2.^(-1 : 4)];  % L2 reg. parameters that we want to try
I = eye(n);

%% set up the final hold-out set
[finalHoldoutIdx, K_full] = setupCVBlocks(size(X,1), CVBSize);


%% loop over all final hold out set
for k = 1 : K_full;
    
    % hold out 1 chunk as the final test set
    X_part = X(~finalHoldoutIdx(:,k),:);
    y_part = y(~finalHoldoutIdx(:,k));
    X_final = X(finalHoldoutIdx(:,k),:);
    y_final = y(finalHoldoutIdx(:,k));
    
    %% setup tuning sets cross-validation blocks for the tunning procedure
    [holdoutIdx, K_tune] = setupCVBlocks(size(X_part,1), CVBSize);
    
    %% tune the parameter: the rank of the truncated SVD
    % preallocate
    BETA = cell(length(LAMBDAS),1);
    TEST_ACCURACY = cell(length(LAMBDAS),1);
    
    for l = 1:length(LAMBDAS)
        lambda = LAMBDAS(l);
        
        % preallocate
        beta = nan(n,K_tune);
        testAccuracy = nan(1,K_tune);
        
        % loop over tunning set (K_part)
        for i = 1:K_tune
            % split the data into training and test set
            X_train = X_part(~holdoutIdx(:,i),:);
            y_train = y_part(~holdoutIdx(:,i));
            X_test = X_part(holdoutIdx(:,i),:);
            y_test = y_part(holdoutIdx(:,i));
            
            %% fit the model
            % compute SVD
            [U,S,V] = svd(X_train, 'econ');
            % compute the beta with U,S,V
            beta(:,i) = V * inv(S^2 + I*lambda) * S * U' * y_train;
            % compute the beta with the normal equations
            beta_test = inv(X_train' * X_train + I*lambda) * X_train' * y_train;
            % check if they are the same
            if any(beta_test - beta(:,i) > 1e-8)
                warning('WARNING: SVD solution and normal equations solution are different!')
            end
            
            % make the prediction
            predict = ones(CVBSize,1);
            predict(X_test * beta(:,i) <= 0) = -1;
            
            %% evaluate the performance
            correctPred = bsxfun(@eq, predict, y_test);
            testAccuracy(i) = sum(correctPred) / CVBSize;
        end
        
        % save the beta and the accuracy by the rank
        BETA{l} = beta;
        TEST_ACCURACY{l} = testAccuracy;
        
    end
    
    %% choose the best parameter
    % compute the mean CV accuracy of a given rank
    accuracyByLambda = mean(cell2mat(TEST_ACCURACY),2);
    % use the rank with the maximal accuracy.
    % if multiple ranks achieves the same accuracy, use the smallest rank
    bestLambdaIdx = find(accuracyByLambda == max(accuracyByLambda) ,1);
    finallambda = LAMBDAS(bestLambdaIdx);
    
    %% fit the final model
    % compute SVD
    [U,S,V] = svd(X_part, 'econ');
    % compute the beta with U,S,V
    beta = V * inv(S^2 + I*finallambda) * S * U' * y_part;
    
    % make the prediction
    predict = ones(CVBSize,1);
    predict(X_final * beta <= 0) = -1;
    
    %% evaluate the performance
    correctPred = bsxfun(@eq, predict, y_final);
    finalAccuracy = sum(correctPred) / CVBSize;
    
    fprintf('Final holdout %d: The "best lambda" = %d, mean accuracy = %f\n', k, bestLambdaIdx, finalAccuracy);
end

end


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
