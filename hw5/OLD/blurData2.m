function [record] = blurData2()
%% load data
load('blurringData.mat')

%% set parameters
[m,n] = size(X);
CVBSize = 53;
LAMBDAS = [0 2.^(-1 : 4)];  % L2 reg. parameters that we want to try
RANK = [1 20:20:500];

%% set up the indices for the final hold out sets 
[finalHoldoutIdx, K_full] = setupCVBlocks(size(X,1), CVBSize);

%% loop over all possible final hold out set (8 of them)
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
        
        
        %% fitting standard LS
%         beta.ls = inv(X_train' * X_train) * X_train' * y_train;
%         finalDev.ls = X_final * beta.ls - y_final;
%         record.ls.finalDev1norm(i,k) = norm(finalDev.ls,1);
        
        %% Truncated SVD
        % compute the SVD
        [U,S,V] = svd(X_train, 'econ');
        for p = 1:length(RANK);
            rank = RANK(p);
            %% fit the model
            S_inv_truncate = truncateS(inv(S), rank);
            % compute the beta with U,S,V
            beta_svd(:,p) = V * S_inv_truncate * U' * y_train;
            % make the prediction
            deviation.svd = X_test * beta_svd(:,p) - y_test;
            % save 1 norm of deviation (parameter by CV block)
            dev1norm.svd(p) = norm(deviation.svd,1);
        end
        
        %% fit the final model with the best rank
        % find the best rank
        bestParam = find(dev1norm.svd == min(dev1norm.svd) ,1, 'first');
        % generate predictions
        finalDev.svd = X_final * beta_svd(:,bestParam) - y_final;
        % save the results
        record.svd.finalDev1norm(i,k) = norm(finalDev.svd,1);
        record.svd.bestRank = RANK(bestParam);
        
        %% fit ridge regression
        [U,S,V] = svd(X_train, 'econ');
        I = eye(size(S));
        for l = 1:length(LAMBDAS)
            % choose a lambda value
            lambda = LAMBDAS(l);
            %% fit the model
            % compute the beta with U,S,V
            beta_ridge(:,l) = V * inv(S^2 + I*lambda) * S * U' * y_train;
            % make the prediction
            deviation.rid = X_test * beta_ridge(:,l) - y_test;
            % save 1 norm of deviation (parameter by CV block)
            dev1norm.rid(l) = norm(deviation.rid,1);
        end
        
        %% check the best weights using the final hold out set
        % find the best rank
        bestLambdaIdx = find(dev1norm.rid == max(dev1norm.rid) ,1, 'first');
        finalDev.ridge = X_final * beta_ridge(:,bestLambdaIdx) - y_final;        
        % save the results
        record.ridge.finalDev1norm(i,k) = norm(finalDev.ridge,1);
        record.ridge.bestLambda = LAMBDAS(bestLambdaIdx);
    
    end % end of the tunning set 
     
end % end of final hold out set 


mean(record.svd.finalDev1norm(:))
mean(record.ridge.finalDev1norm(:))

end % end of the main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% set up the cv blocks
function [holdOutIdx, numFolds] = setupCVBlocks(numData, cvBlockSize)
numFolds = round(numData/cvBlockSize);
holdOutIdx = false(numData,numFolds);        % the indicies for the hold out set
% get the hold out set indices for K folds
for i = 1 : numFolds
    holdOutIdx(((i-1)*cvBlockSize+1):(i)*cvBlockSize,i) = true;
end

% trim the useless chuck
if size(holdOutIdx,1) ~= numData
    holdOutIdx = holdOutIdx(1:numData,:);
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