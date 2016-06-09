%% compare 2 kernel methods 
function [accuracy] = compare3methods(trainingSetSize)
% generate training and testing data
[A, b] = genTraining_data(trainingSetSize, 0);
[Atest, btest] = genTraining_data(100, 0);
% set lambda
lambda = 1e-5;

%% compute the polynomial kernal
m = size(A,1);
ker.poly = nan(m,m);
ker.gaus = nan(m,m);
for i=1:m
    for j=1:m
        ker.poly(i,j) = (dot(A(i,:), A(j,:)) +1 )^2;
        ker.gaus(i,j) = exp(-.5 * (norm(A(i,:) - A(j,:),2))^2);
    end
end

%% fit the kernelized least square
alpha.poly = inv(ker.poly + lambda * eye(size(ker.poly))) * b;
alpha.gaus = inv(ker.gaus + lambda * eye(size(ker.gaus))) * b;

%% compute classification accuracy
prediction.poly = nan(size(Atest,1),1);
prediction.gaus = nan(size(Atest,1),1);
% compute the kernel LS prediction 1 by 1
for i = 1 : size(Atest,1)
    xnew = Atest(i,:);
    
    % compute kernel LS prediction for 1 example (polynomial)
    temp = 0; 
    for j = 1 : m
        temp = temp + alpha.poly(j) * (dot(xnew, A(j,:)) +1 )^2;
    end
    prediction.poly(i) = sign(temp);
    
    % compute kernel LS prediction for 1 example (gaussian)
    temp = 0;
    for j = 1 : m
        temp = temp + alpha.gaus(j) * exp(-.5 * (norm(xnew - A(j,:),2) )^2);
    end
    prediction.gaus(i) = sign(temp);
    
end

%% fit standard LS
wts.ls = pinv(A)*b;
prediction.ls = sign(Atest * wts.ls);


%% compute accuracy for the 2 methods
accuracy.p = sum(prediction.poly == btest) / length(btest);
accuracy.g = sum(prediction.gaus == btest) / length(btest);
accuracy.ls = sum(prediction.ls == btest) / length(btest);
end
