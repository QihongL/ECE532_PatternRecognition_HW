% initalization 
close all; clear all; clc;
load fisheriris.mat
X = meas;
numData = 50;
y = [zeros(numData,1); ones(numData,1); 2*ones(numData,1)];
[m,n] = size(meas);
I = eye(n,n);

% fit standard LS
beta_hat = inv(X' * X) * X' * y;

% set learning rate
upperLearningRate = 2 / norm(X,2)^2;
lr = upperLearningRate/1.1;

%% land weber iteration
beta = zeros(n,1);
error = zeros(n,1);
diff(1) = norm(beta_hat - beta,2);
i = 1; 
while diff(i) > 1e-5
    i = i + 1; 
    % update weight 
    beta = beta - lr * X' * (X * beta - y);
    % record difference from truth 
    diff(i) = norm(beta_hat - beta,2);
    % record error 
    error(:,i) = beta - beta_hat;
    if i > 100000
        break;
    end
end

plot(diff)
i
[beta beta_hat]

k = 3
[(I - lr * X' * X) * error(:,k) error(:,k + 1)]