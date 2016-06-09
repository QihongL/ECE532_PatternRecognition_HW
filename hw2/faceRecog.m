%% Homework 2 - Question 5.e - emotion recognition - cross validation 
function faceRecog()
load('face_emotion_data.mat')
%% feature selction 
% X = X(:,[1 3 4]);

%% get some data parameters 
m = size(X,1);      % num of data
n = size(X,2);      % num features
K = 8;              % folds of CV
% set up the cross validation blocks 
[holdOutIdx, cvBlockSize] = setupCVBlocks(K, m);

%% fit the OLS model, with cross validation 
beta = nan(n,K);        % preallocate beta (features by K)
testAcc = nan(1,K);     % a test set accuracy for each cv block
testDev = nan(1,K);     % deviation from test set label for each cv block
prediction = zeros(cvBlockSize,K); 
% loop over all cv blocks 
for i = 1:K 
    % select the appropriate subset of the data
    Xtrain = X(~holdOutIdx(:,i),:);
    ytrain = y(~holdOutIdx(:,i));
    Xtest = X(holdOutIdx(:,i),:);
    ytest = y(holdOutIdx(:,i));
    
    % fit the model using normal equation 
    beta(:,i) = inv(Xtrain' * Xtrain) * Xtrain' * ytrain;
    
    % make the prediction based on if it is close to 1 or -1 
    prediction(Xtest * beta(:,i) > 0, i) = 1;
    prediction(Xtest * beta(:,i) <=0, i) = -1;
    
    % compare the predictions with the test set labels
    correctPredictions = bsxfun(@eq, prediction(:,i), ytest);
    % compare the accuracy on the test set
    testAcc(i) = sum(correctPredictions) / cvBlockSize;
    testDev(i) = sum(abs(Xtest * beta(:,i) - ytest));
end


%% print the cv accuracy
printPerformance(testAcc, testDev)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [holdOutIdx, cvBlockSize] = setupCVBlocks(numFolds, numData)
%% set up the cv blocks
cvBlockSize = numData/numFolds;              % assume this is divisible
holdOutIdx = false(numData,numFolds);        % the indicies for the hold out set
% get the hold out set indices for K folds
% fprintf('The following %d-folds CV blocks were created:\n', numFolds)
for i = 1 : numFolds
%     fprintf('%-4d to %-4d\n',(i-1)*cvBlockSize+1, (i)*cvBlockSize)
    holdOutIdx(((i-1)*cvBlockSize+1):(i)*cvBlockSize,i) = true;
end
end

function printPerformance(testAcc, testDev)
% compute the the mean accuracy on the test set
fprintf('\nCross validated performance on the corresponding test set:\n');
fprintf('\t\t');
for i = 1 : length(testAcc);
    fprintf('CV%.2d\t',i);
end
fprintf('\n cvAccuracy:   ');
% print the cv accuracy
for i = 1 : length(testAcc);
    fprintf('%-8.4f',testAcc(i));
end
% fprintf('\n absDeviation: ');
% % print the sum of abs deviation 
% for i = 1 : length(testAcc);
%     fprintf('%-8.4f',testDev(i));
% end
% print the mean accuracy
fprintf('\n\nMean accuracy : %-8.4f\n', mean(testAcc));
% fprintf('Mean deviation: %-8.4f\n', mean(testDev));
end
