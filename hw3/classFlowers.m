%% ECE 532 - HW3 - Question 4
function classFlowers()
% initialization
clear all; clc
load fisheriris.mat
selectFirstThreeFeatures = 0;
projectOnTo2DSpace = 1; 
if projectOnTo2DSpace && ~selectFirstThreeFeatures
    warning('My projection code is based on the use of 3 features');
    selectFirstThreeFeatures = 1;
end

% some parameters
numData = 50;
% set the number of data points in the training data (< 50)
maxTrainSize = numData - 1;
% choose features (e.g. 1110 meas select the first three features)
if selectFirstThreeFeatures
    featureSelector = logical([1 1 1 0])
    % read the design matrix
    X = meas(:,featureSelector);
    if projectOnTo2DSpace
        % principle component analysis 
        pcs = pca(X);
        % use the subspace spanned by the 1st two PCs
        disp('Here are the basis for the lower dimensional subspace, where the data were projected')
        S = pcs(:,1:2)
        % calculate the projection matrix 
        projMat = S*inv(S'*S)*S';
        % do orthogonal projection 
        projectedX = X * projMat;
        X = projectedX(:,1:2);
    end
else
    % no feature selection & no projection
    X = meas;
end
% 0 = setosa
% 1 = versicolor
% 2 = virginica
y = [zeros(numData,1); ones(numData,1); 2*ones(numData,1)];

%% Run the analysis
% conduct multiple tests
numTrials = 50;
accuracy = nan(numTrials,maxTrainSize);
for trainSize = 1: maxTrainSize;
    for i = 1 : numTrials
        %% create vectors that randomly select training data
        holdOutIdx = getTrainingSetIndex(numData,trainSize);
        
        %% split the data into training and testing set
        Xtrain = X(holdOutIdx,:);
        ytrain = y(holdOutIdx);
        Xtest  = X(~holdOutIdx,:);
        ytest  = y(~holdOutIdx);
        
        %% fit the LS model
        beta = inv(Xtrain' * Xtrain) * Xtrain' * ytrain;
        
        %% compute the accuracy
        predictions = round(Xtest * beta);
        predictions = transformPredictions(predictions, y);
        accuracy(i,trainSize) = sum(predictions == ytest) / length(ytest);
    end
    
end
% compute the error
error = 1 - accuracy;
fprintf('Test set accuracy when using training set size of 40 is %f \n', mean(error(:,40)));

%% plot the performance 
plotPerformance_trainSize(error)

% plot the first three features in the scatter plot if we only use the 1st
% three features
if selectFirstThreeFeatures
    disp('Press something to see the scatter plots!'); pause;
    threeFeatureScatterPlots(meas)
    % if also project onto 2D space, plot the projected 2D scatter plot 
    if projectOnTo2DSpace
        disp('Press something to see the projected plots!'); pause;
        projectedScatterPlots(projectedX,X)
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% create vectors that randomly select the holdout set of given size
function holdOutIdx = getTrainingSetIndex(numData,trainSize)
% check input
if trainSize < 1
    error('Training size should be at least 1');
elseif trainSize >= numData
    error('Training size should be smaller than numData');
end
% preallocate
trainIdx.set = false(numData, 1);
trainIdx.ver = false(numData, 1);
trainIdx.vir = false(numData, 1);
% randomly choose some of them
trainIdx.set(randperm(numData, trainSize)) = true;
trainIdx.ver(randperm(numData, trainSize)) = true;
trainIdx.vir(randperm(numData, trainSize)) = true;
% concatenate the index vector
holdOutIdx = [trainIdx.set; trainIdx.ver; trainIdx.vir];
end


%% handle impossible predictions
% the range of y is {0, 1, 2}, so if the classifer under/over shot, force
% it to be the closest label in the range
function predictions = transformPredictions(predictions, y)
% this is written in a more genearl form that is not specific to {0,1,2}
% range
ymax = max(y);
ymin = min(y);
predictions(predictions > ymax) = ymax;
predictions(predictions < ymin) = ymin;
end


%% Plot the performance against trainSize
function plotPerformance_trainSize(error)
FS = 16;
% plot the error vector, which records the error by training set size
plot(mean(error),'LineWidth',2)
% add some descriptions
ylim([0,1])
title('Classification performance against the training set size', 'fontsize', FS)
xlabel('Number of data points used in the training set', 'fontsize', FS)
ylabel('Mean CV error (%)', 'fontsize', FS)
end

%% create 3 d scatter plots for 3 features
function threeFeatureScatterPlots(meas)
FS = 16;

% create the scatter plots
subplot(1,2,1)
scatter3(meas(:,1),meas(:,2),meas(:,3))
title('Scatter plot for the 1st three features', 'fontsize', FS)
xlabel('x - sepal length', 'fontsize', FS)
ylabel('y - sepal width', 'fontsize', FS)
zlabel('z- petal length', 'fontsize', FS)

% create the SAME scatter plots (for rotating)
subplot(1,2,2)
scatter3(meas(:,1),meas(:,2),meas(:,3))
title('Scatter plot for the 1st three features (for rotating)', 'fontsize', FS)
xlabel('x - sepal length', 'fontsize', FS)
ylabel('y - sepal width', 'fontsize', FS)
zlabel('z- petal length', 'fontsize', FS)

end

%% plot the projected 2D scatter plot 
function projectedScatterPlots(projectedX,X)
FS = 16;
% create the PROJECTED scatter plots
subplot(3,1,1)
scatter3(projectedX(:,1),projectedX(:,2),projectedX(:,3))
title('Projected plot', 'fontsize', FS)
xlabel('x - sepal length', 'fontsize', FS)
ylabel('y - sepal width', 'fontsize', FS)
zlabel('z- petal length', 'fontsize', FS)

% create the PROJECTED scatter plots
subplot(3,1,2)
scatter3(projectedX(:,1),projectedX(:,2),projectedX(:,3))
title('Projected plot (for rotating)', 'fontsize', FS)
xlabel('x - sepal length', 'fontsize', FS)
ylabel('y - sepal width', 'fontsize', FS)
zlabel('z- petal length', 'fontsize', FS)

% create the PROJECTED scatter plots (for rotating)
subplot(3,1,3)
plot (X(:,1), X(:,2),'o');
title('Projected plot (view in 2D space)', 'fontsize', FS)
xlabel('First PC', 'fontsize', FS)
ylabel('Second PC', 'fontsize', FS)

end
