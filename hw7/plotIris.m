%% ECE 532 - HW7
clear all;clc;close all;
% initialization
load fisheriris.mat

% some parameters
numData = 50;
% no feature selection & no projection
X = meas(51:end,3:4);
X = [ones(100,1), X];
% versicolor = -1
% virginica  = 1
y = [-1* ones(numData,1); ones(numData,1)];

%% plot the data
hold on
plot(X(y ==-1,2), X(y ==-1,3), 'x')
plot(X(y == 1,2), X(y == 1,3), 'o')


FS = 14;
xlim([min(X(:,2))-1 max(X(:,2))+1])
ylim([min(X(:,3))-1 max(X(:,3))+1])
title('Data visualized in the feature space', 'fontsize', FS)
xlabel('First feature', 'fontsize', FS)
ylabel('Second feature', 'fontsize', FS)


%% fit standard ls
w.ls = inv(X' * X) * X' * y;

% plot the decision boundary for ls.
boundary.f1_range = [min(X(:,2)) :0.01: max(X(:,2))];
boundary.f2_val.ls = -w.ls(1)/w.ls(3)-(w.ls(2)/w.ls(3)) .* boundary.f1_range';
plot(boundary.f1_range, boundary.f2_val.ls);
% legend({'versicolor', 'virginica', 'LS boundary'}, 'fontsize', FS, 'location', 'northwest')

%% fit svm
tau = 0.003;
lambda = .1;
w.svm = zeros(size(X,2),1);
numIters = 20000;
record.svmw = nan(length(w.svm),numIters);
% implement gradient decent
for i = 1 : numIters
    
    change = zeros(3,1);
    % accumulate the gradient for all training examples
    for j = 1 : length(y)
        if y(j) * X(j,:) * w.svm < 1
            change(1) = change(1) - tau * y(j) * X(j,1)';
            change(2:3) = change(2:3) - tau*(y(j) * X(j,2:3)' - 2*lambda * w.svm(2:3));
        end
    end
    % gradient decent update
    w.svm = w.svm - change;
    % save weights
    record.svmw(:,i) = w.svm;
    % 
    disp(i)
    norm(y - X * w.svm)
end

%% plot svm boundaries
boundary.f2_val.svm = -w.svm(1)/w.svm(3)-(w.svm(2)/w.svm(3)) .* boundary.f1_range';
plot(boundary.f1_range, boundary.f2_val.svm)
legend({'versicolor', 'virginica','LS boundary', 'SVM boundary'},'fontsize', FS, 'location', 'northwest')
hold off
%% plot weights
figure
subplot(3,1,1)
plot(record.svmw(1,:))
title_text = sprintf('Weights over time, learning rate = %.5f', tau);
title(title_text,'fontsize', FS)
ylabel('slope','fontsize', FS)
subplot(3,1,2)
plot(record.svmw(2,:))
ylabel('1st feature','fontsize', FS)
subplot(3,1,3)
plot(record.svmw(3,:))
ylabel('2rd feature','fontsize', FS)
