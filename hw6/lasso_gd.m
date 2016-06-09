%% fit lasso using gradient decent
function [finalbeta, history] = lasso_gd(X,y, lambda, tau, display)
% interation constants
[m,n] = size(X);
MAX_ITER = 10000;
TOLERANCE = 1e-3;

% preallocate
beta = zeros(n,1);
% precompute some unchanging things
XTy = X' * y;
XTX = X' * X;
for i = 1 : MAX_ITER
    % update
    grad = - 2 * XTy + 2 * XTX * beta(:,i) + lambda * sign(beta(:,i));
    beta(:,i+1) = beta(:,i) -  tau * grad;
    
    %% Performance monitoring
    % compute change in beta
    diff = norm(beta(:,i+1) - beta(:,i));
    % nnz for the updated beta
    nnzb = n-numZeros(beta(:,i+1));
    % error of the updated beta
    history.accuracy(i) = sum(sign(X * beta(:,i+1)) == y) / m;
    % print the performance
    if display
        fprintf('%4d%12.4f %16.4f %10d\n', i, history.accuracy(i), diff, nnzb);
    end
    if diff < TOLERANCE
        break;
    end
end
% save history and final parameters
history.beta = beta;
finalbeta = beta(:,end);
end

%% check the number of non zero betas
function [nz] = numZeros(beta)
tolerance = 1e-6;
% calculate number of "zeros"
nz = sum(abs(beta) <= tolerance);
end
