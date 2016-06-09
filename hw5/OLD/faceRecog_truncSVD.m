function [] = faceRecog_truncSVD()
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
[m,n] = size(X);    % 
CVBSize = 16;       % cross validation block size (diviside by m)

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
    maxRank = n; 
    BETA = cell(maxRank,1);
    TEST_ACCURACY = cell(maxRank,1);
    
    for rank = 1:maxRank;
        % preallocate
        beta = nan(maxRank,K_tune);
        testAccuracy = nan(1,K_tune);
        
        % loop over tunning set (K_part)
        for i = 1:K_tune
            % split the data into training and test set
            X_train = X_part(~holdoutIdx(:,i),:);
            y_train = y_part(~holdoutIdx(:,i));
            X_test = X_part(holdoutIdx(:,i),:);
            y_test = y_part(holdoutIdx(:,i));
            
            %% fit the model
            % compute the singular value decomposition
            [U,S,V] = svd(X_train, 'econ');
            S_inv_truncate = truncateS(inv(S), rank);
            % compute the beta with U,S,V
            beta(:,i) = V * S_inv_truncate * U' * y_train;
            % make the prediction
            predict = ones(CVBSize,1);
            predict(X_test * beta(:,i) <= 0) = -1;
            
            %% evaluate the performance
            correctPred = bsxfun(@eq, predict, y_test);
            testAccuracy(i) = sum(correctPred) / CVBSize;
        end
        
        % save the beta and the accuracy by the rank
        BETA{rank} = beta;
        TEST_ACCURACY{rank} = testAccuracy;
        
    end
    
    %% choose the best parameter
    % compute the mean CV accuracy of a given rank 
    accuracyByRank = mean(cell2mat(TEST_ACCURACY),2);
    % use the rank with the maximal accuracy. 
    % if multiple ranks achieves the same accuracy, use the smallest rank
    bestRank = find(accuracyByRank == max(accuracyByRank) ,1);
    
    %% fit the final model
    % compute the singular value decomposition
    [U,S,V] = svd(X_part, 'econ');
    S_inv_truncate = truncateS(inv(S), bestRank);
    % compute the beta with U,S,V
    beta = V * S_inv_truncate * U' * y_part;
    % make the prediction
    predict = ones(CVBSize,1);
    predict(X_final * beta <= 0) = -1;
    
    %% evaluate the performance
    correctPred = bsxfun(@eq, predict, y_final);
    finalAccuracy = sum(correctPred) / CVBSize;
    
    fprintf('Final holdout %d: The "best rank" = %d, mean accuracy = %f\n', k, bestRank, finalAccuracy);
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
