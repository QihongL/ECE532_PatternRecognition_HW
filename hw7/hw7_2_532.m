%% initialization - simulation parameters
close all; clear; clc;
m = 10;     % num data points
w.true = [0 ;1];
lapNoise = 1; 


%% simulate the data
% generate data X ~ unif[0,1]
x = unifrnd(0,1,m,1);
x = [ones(m,1) x];
% Gaussian noise
if ~lapNoise
    err = randn(m,1)/5;
else 
    err = laprnd(m,1)/5;
end
% then generate y
y = x * w.true + err;

%% LS fit
w.ls = inv(x' * x) * x' * y;


%% l1 loss - gradient decent 
% set learning rate 
tau = 0.001;

% preallocate 
numIters = 1000;
w.gd = zeros(size(w.true));
w.sgd = zeros(size(w.true));
residual.sgd = nan(numIters,1);
% iterative alg.
for j = 1 : numIters
    %% random permutation stochastic decent procedure
    for i = randperm(m)
        % compute the stochastic gradient for 1 training example 
        if (y(i) - x(i,:) * w.sgd) > 0
            sgd = x(i,:)';
        elseif (y(i) - x(i,:) * w.sgd) < 0
            sgd = -x(i,:)';
        else
            sgd = zeros(size(w.true));  % sub grad
        end
        % stochastic gradient decent update
        w.sgd = w.sgd + tau * sgd;
    end
    
    
    %% gradient decent procedure
    gd = zeros(size(w.true));
    for i = 1:m
        % compute the gradient for 1 training example 
        if (y(i) - x(i,:) * w.sgd) > 0
            sgd = x(i,:)';
        elseif (y(i) - x(i,:) * w.sgd) < 0
            sgd = -x(i,:)';
        else
            sgd = zeros(size(w.true));  % sub grad
        end
        % accumulate gradients 
        gd = gd + sgd;
    end
    % gradient decent update
    w.gd = w.gd + tau * gd;
    
    %% record the residual
    residual.sgd(j) = norm(y - x * w.sgd,1);
    residual.gd(j) = norm(y - x * w.gd,1);
end


%%
[w.true w.ls w.sgd w.gd]

%% plot the data and decision boundary
hold on 
plot(x(:,2), y, 'x')
boundary.x = min(x(:,2)) :0.01: max(x(:,2));
boundary.y.ls = w.ls(1) + w.ls(2) * boundary.x;
boundary.y.gd = w.gd(1) + w.gd(2) * boundary.x;
boundary.y.sgd = w.sgd(1) + w.sgd(2) * boundary.x;
plot(boundary.x, boundary.y.ls, 'linewidth', 2)
plot(boundary.x, boundary.y.gd, 'linewidth', 2)
plot(boundary.x, boundary.y.sgd, 'linewidth', 2)
hold off

% add some text 
FS = 14; 
xlabel('X','fontsize', FS)
ylabel('y','fontsize', FS)
title('the data and the decision boundaries','fontsize', FS)
legend({'data', 'l2 loss boundary', 'l1 loss boundary (GD)', 'l1 loss boundary (SGD)'},'fontsize', FS, 'location', 'northwest')

%% plot residual over time 
figure
hold on 
plot(residual.sgd, 'linewidth', 2)
plot(residual.gd', 'linewidth', 2)
hold off
legend({'stochastic gradient decent', 'gradient decent'},'fontsize', FS, 'location', 'northwest')
xlabel('iterations','fontsize', FS)
ylabel('residual - 1 norm','fontsize', FS)
title('residual over time','fontsize', FS)