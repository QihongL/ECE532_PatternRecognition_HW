%% signal reconstruction
function blurDataAnalysis(k, sigma)
% load the data
if nargin == 0
    k = 30; 
    sigma = 0.01; 
end
[X, y, btrue] = generateBluringData(k,sigma, false);
%% set the choice of parameter 
LAMBDAS = [0 2.^(-13 : 11)];
RANK = [1 20:20:500];
normType = 2; 

%% standard LS
beta.ls = inv(X' * X) * X' * y;
% make the prediction
prediction.ls = X * beta.ls;
deviation.ls = y - prediction.ls;
% save 1 norm of deviation (parameter by CV block)
deviationY_1norm.ls = norm(deviation.ls,normType);
deviationB_1norm.ls = norm(beta.ls - btrue,normType);

%% truncated SVD
[U,S,V] = svd(X, 'econ');
for p = 1:length(RANK);
    rank = RANK(p);
    % fit the model
    S_inv_truncate = truncateS(inv(S), rank);
    % compute the beta with U,S,V
    beta.svd(:,p) = V * S_inv_truncate * U' * y;
    % make the prediction
    prediction.svd(:,p) = X * beta.svd(:,p);
    deviation.svd = y - prediction.svd(:,p);
    % save 1 norm of deviation (parameter by CV block)
    deviationY_1norm.svd(p) = norm(deviation.svd,normType);
    deviationB_1norm.svd(p) = norm(beta.svd(:,p) - btrue, normType);
end

%% RLS
I = eye(size(S));
for l = 1:length(LAMBDAS)
    % choose a lambda value
    lambda = LAMBDAS(l);
    % compute the beta with U,S,V
    beta.ridge(:,l) = V * inv(S^2 + I*lambda) * S * U' * y;
    % make the prediction
    prediction.ridge(:,l) = X * beta.ridge(:,l);
    deviation.rid = y - prediction.ridge(:,l);
    % save 1 norm of deviation (parameter by CV block)
    deviationY_1norm.rid(l) = norm(deviation.rid,normType);
    deviationB_1norm.rid(l) = norm(beta.ridge(:,l) - btrue, normType);
end


%% plot the performance 
% close all; 
subplot(2,1,1)
plotDeviation(deviationY_1norm, 'deviation of y', k, sigma)
subplot(2,1,2)
plotDeviation(deviationB_1norm, 'deviation of beta', k, sigma)


%% visualize predicted betas against the truth
figure;
plotBetaDeviation(deviationB_1norm, beta, btrue, normType);
figure;
plotBetaDeviation(deviationY_1norm, prediction, y, normType);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot predicted beta against truth
function plotBetaDeviation(deviation, beta, btrue, normType)
FS = 14;
ylab_text = sprintf('Deviation (%d norm)', normType);
% find the index for the "best beta"
bestRidgeIdx = deviation.rid == min(deviation.rid);
bestSvdIdx = deviation.svd == min(deviation.svd);

% plot them 
subplot(3,1,1)
plot([beta.ls btrue])
legend({'Reconstructed Signal', 'True Signal'}, 'fontsize', FS, 'location', 'southeast')
title('Standard least square', 'fontsize', FS)
ylabel(ylab_text, 'fontsize', FS)
subplot(3,1,2)
plot([beta.svd(:,bestSvdIdx) btrue])
legend({'Reconstructed Signal', 'True Signal'}, 'fontsize', FS, 'location', 'southeast')
title('Truncated SVD', 'fontsize', FS)
ylabel(ylab_text, 'fontsize', FS)
subplot(3,1,3)
plot([beta.ridge(:,bestRidgeIdx) btrue])
legend({'Reconstructed Signal', 'True Signal'}, 'fontsize', FS, 'location', 'southeast')
title('L2 regularized least square', 'fontsize', FS)
ylabel(ylab_text, 'fontsize', FS)
xlabel('feature indices', 'fontsize', FS)

end

%% function: plot deviation on y
function [] = plotDeviation(dev1norm, ylabel_text, k, sigma)
LWD = 1.5;
FS = 14;
title_text = sprintf('Compare 3 models  (k = %d, Sigma = %.3f)', k, sigma);
hold on
plot(dev1norm.svd, 'linewidth', LWD)
plot(dev1norm.rid, 'linewidth', LWD)
plot(xlim,[dev1norm.ls dev1norm.ls], 'linewidth', LWD)
plot(xlim,5)
hold off
legend({'truncated SVD', 'L2 regularization', 'Standard LS'}, ...
    'fontsize', 14, 'location', 'northwest')
title(title_text, 'fontsize', FS)
ylabel(ylabel_text, 'fontsize', FS)
xlabel('Different parameters', 'fontsize', FS)
end

%% truncate the diagnal matrix S: leave the first r diagnal entries
function S_truncate = truncateS(S, rank)
% truncate singular values
S_truncate = S;
for i = rank+1 : length(diag(S))
    S_truncate(i,i) = 0;
end
end