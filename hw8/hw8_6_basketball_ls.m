%% basket ball example 
clear all; clc; close all
A = [1 1 1 1; 1.78 1.8 1.85 2.08]';
b = [-1 -1 1 1]';

svm = svmtrain(A,b,'showplot', 0,'kernel_function','linear')
prediction = svmclassify(svm,A(3,:),'ShowPlot',0)

%% fitting LS
x = inv(A' * A) * A' * b;

% it is wrong for the 3rd guy, because th 4th guy is pull the decision
% boundary towards it!
sign(A * x)


%% fitting SVM
tau = 0.01; 
lambda = 0; 
w_svm = zeros(2,1);

%% one update 
change = zeros(2,1);
for i = 1 : size(A,1)
    if b(i) * A(i,:) * w_svm > 1
        delta = 0; 
    else
        delta = -b(i) * A(i,:)';
    end
    delta
    change = change - tau * (delta + 2 * lambda * w_svm);
end
w_svm = w_svm - change;
[w_svm x]

%% plot the decision boundary 
boundary = -x(1)/x(2);
plot(boundary, 'rx', 'linewidth', 2)
title('Least square fit' ,'fontsize', 14)
ylabel('height', 'fontsize', 14)
legend({'LS Decision boundary'} ,'fontsize', 14)
ylim([min(A(:,2))-.1 max(A(:,2))+.1])
hold on
for i = 1 : size(A,1)
    if b(i) >0
        plot(A(i,2), 'bo', 'linewidth', 2)
    else
        plot(A(i,2), 'go', 'linewidth', 2)
    end
end

hold off