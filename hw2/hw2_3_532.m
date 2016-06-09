%% Homework 2 - Question 3 - fitting d degree polynomial 
clear all; clc;clf;
load('polydata.mat')
hold on 

% plot the data
scatter(a,b);
m = length(a);

for d = 1 : 3   % fit 1,2 and 3 degree polynomial

% set up the design matrix
X = nan(m,d + 1);
for i = 0 : d 
	X(:,i+1) = a.^i;
end
% find the weights
beta = inv(X'*X)* X'*b;

% visualize the results
% set the range for the fitted curve
ranges = min(a) :0.01: max(a);
ranges = ranges';
RANGES = nan(length(ranges),d + 1);
for i = 0 : d 
	RANGES(:,i+1) = ranges.^i;
end
% evaluate the prediction at every point
predictions = RANGES * beta;

% plot the curve
plot(ranges,predictions,'LineWidth',2)
end

% attach title, legends
FS = 14; 
legend({'data', 'd = 1','d = 2','d = 3'},'FontSize',FS,'Location','northwest')
tt = sprintf('Fitting results for degree-d polynomials');
title(tt,'fontsize',FS)

hold off; 