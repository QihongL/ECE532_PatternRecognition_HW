function accuracy = compare3kernels_svm(m)
% generate training and test data
% m = 100;
m_test = 100;
[A,b] = genTraining_data(m,false);
[Atest,btest] = genTraining_data(m_test,false);

%% compute svm with rbf or poly kernel
% subplot(1,2,1)
svm.rbf = svmtrain(A,b, 'ShowPlot', 0, 'kernel_function', 'rbf');
% subplot(1,2,2)
svm.poly = svmtrain(A,b, 'ShowPlot', 0, 'kernel_function', 'polynomial');

%% compute the accuracy for rbf and poly kernelized svm
prediction.rbf = nan(m_test,1);
prediction.poly = nan(m_test,1);
for i = 1 : m_test
    Xnew = Atest(i,:);
    prediction.rbf(i) = svmclassify(svm.rbf,Xnew,'ShowPlot',0);
    prediction.poly(i) = svmclassify(svm.poly,Xnew,'ShowPlot',0);
end
% compute accuracy
accuracy.rbf = sum(prediction.rbf == btest) / length(btest);
accuracy.poly = sum(prediction.poly == btest) / length(btest);

%% fit standard least square model and compute the accuracy
wts.ls = pinv(A) * b;
prediction.ls = sign(Atest*wts.ls);
accuracy.ls = sum(prediction.ls == btest)/length(btest);
end