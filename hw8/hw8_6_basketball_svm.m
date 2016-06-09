clear all;close all;clc;
X = [1 1 1 1; 1.78 1.8 1.85 2.08]';
y = [-1 -1 1 1]';

%% fitting LS
w.ls = inv(X' * X) * X' * y;
sign(X * w.ls)

%% fit svm
tau = 0.03;
lambda  = 0;
w.svm = zeros(size(X,2),1);
numIters = 100000;
% implement gradient decent
for i = 1 : numIters
    change = zeros(size(X,2),1);
    % accumulate the gradient for all training examples
    for j = 1 : length(y)
        if 1 - y(j) * X(j,:) * w.svm > 0 
            change(:) = change(:) - y(j) * X(j,:)';
        end
    end
    % gradient decent update
    w.svm = w.svm - tau * change;
    % 
    fprintf('%d %f \n', i, norm(y - X * w.svm));    
end

%% plot the decision boundary 
hold on
plot(-w.ls(1)/w.ls(2), 'rx', 'linewidth', 2)
plot(-w.svm(1)/w.svm(2), 'mx', 'linewidth', 2)

title('Compare 2 decision boundaries' ,'fontsize', 14)
ylabel('height', 'fontsize', 14)
legend({'LS Decision boundary', 'SVM boundary'} ,'fontsize', 14)
ylim([min(X(:,2))-.1 max(X(:,2))+.1])
% plot the points 
for i = 1 : size(X,1)
    if y(i) >0
        plot(X(i,2), 'bo', 'linewidth', 2)
    else
        plot(X(i,2), 'go', 'linewidth', 2)
    end
end
hold off
