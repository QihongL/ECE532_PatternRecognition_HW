clear all; close all; clc
sampleSize = 100;

% compare mean accuracy for small training set size
accuracies.small = nan(sampleSize,3);
for i = 1 : sampleSize
    accuracy = compare3kernels_svm(10);
    accuracies.small(i,:) = struct2array(accuracy);
end

% compare mean accuracy for small training set size
accuracies.mid = nan(sampleSize,3);
for i = 1 : sampleSize
    accuracy = compare3kernels_svm(100);
    accuracies.mid(i,:) = struct2array(accuracy);
end

% compare mean accuracy for small training set size
accuracies.big = nan(sampleSize,3);
for i = 1 : sampleSize
    fprintf('%d\n',i);
    accuracy = compare3kernels_svm(1000);
    accuracies.big(i,:) = struct2array(accuracy);
end
save('accuracies', 'accuracies')


%% compare accuracy 
acc(1,:) = mean(accuracies.small);
acc(2,:) = mean(accuracies.mid);
acc(3,:) = mean(accuracies.big);

acc_std(1,:) = std(accuracies.small);
acc_std(2,:) = std(accuracies.mid);
acc_std(3,:) = std(accuracies.big);


hold on 
errorbar([1 2 3],acc(:,1),acc_std(:,1), 'linewidth',2)
errorbar([1 2 3],acc(:,2),acc_std(:,2), 'linewidth',2)
errorbar([1 2 3],acc(:,3),acc_std(:,3), 'linewidth',2)

legend({'SVM with polynomial kernel', 'SVM with Gaussian kernel', 'least square'}, 'fontsize', 15)
title('Compare SVM with different kernels', 'fontsize', 15)
ylabel('accuracy', 'fontsize', 15)
xlabel('increasing training set size', 'fontsize', 15)
xlim([.8,3.2])
set(gca,'XTickLabel',{'10', ' ', '100', ' ', '1000'})
