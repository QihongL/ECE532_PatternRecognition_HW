clear all; close all; clc; 
% compute the performance for ridge and svd, set this as the baseline
[baseLineAcc] = faceRecog(0);
% compute the performance for ridge and svd when adding 3 redundent features 
sampSize = 10;
accuracy = cell(sampSize,1);
for i = 1 : sampSize
    accuracy{i} = faceRecog(1);
end
meanAcc.svd = []; meanAcc.ridge = [];
for i = 1 : sampSize
    meanAcc.svd = [meanAcc.svd mean(accuracy{i}.svd(:))];
    meanAcc.ridge = [meanAcc.ridge mean(accuracy{i}.ridge(:))];
end

% t test, see if their difference is zero 
[h1,p1,ci1,stats1] = ttest(meanAcc.svd - mean(baseLineAcc.svd(:)))
[h2,p2,ci2,stats2] = ttest(meanAcc.ridge - mean(baseLineAcc.ridge(:)))
