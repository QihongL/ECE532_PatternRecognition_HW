clear all;close all;clc
load('LASSO_BEST.mat')
load('RIDGE_BEST.mat')
K = 10;
for k = 1:K
    
    errors.lasso(k,:) = 1- LASSO_BEST{k}.accuracy;
    errors.ridge(k,:) = 1- RIDGE_BEST{k}.accuracy;
    sqrErrors.lasso(k,:) = LASSO_BEST{k}.diff;
    sqrErrors.ridge(k,:) = RIDGE_BEST{k}.diff;
    
    nnz.lasso(k,:) = sum(abs(LASSO_BEST{k}.beta) > 1e-6);
    nnz.ridge(k,:) = sum(abs(RIDGE_BEST{k}.beta) > 1e-6);
end


mean(errors.lasso(:))
mean(errors.ridge(:))

mean(sqrErrors.lasso(:))
mean(sqrErrors.ridge(:))

mean(nnz.lasso(:))
mean(nnz.ridge(:))


subplot(1,3,1)
barwitherr([std(errors.lasso(:)) std(errors.ridge(:))], [mean(errors.lasso(:)) mean(errors.ridge(:)) ])
set(gca,'XTickLabel',{'lasso','ridge'}, 'fontsize', 15)
title('Classification errors', 'fontsize', 15)

subplot(1,3,2)
barwitherr([std(sqrErrors.lasso(:)) std(sqrErrors.ridge(:))], [mean(sqrErrors.lasso(:)) mean(sqrErrors.ridge(:)) ])
set(gca,'XTickLabel',{'lasso','ridge'}, 'fontsize', 15)
title('Squared errors', 'fontsize', 15)

subplot(1,3,3)
barwitherr([std(nnz.lasso(:)) std(nnz.ridge(:))], [mean(nnz.lasso(:)) mean(nnz.ridge(:)) ])
ylim([0 9000])
set(gca,'XTickLabel',{'lasso','ridge'}, 'fontsize', 15)
title('Number of non-zero weights', 'fontsize', 15)