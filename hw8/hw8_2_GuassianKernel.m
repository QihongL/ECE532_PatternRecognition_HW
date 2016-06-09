clear; close all;clc
training_data
% set lambda
lambda = 1e-5;

%% compute the polynomial kernal
m = size(A,1);
gauKer = nan(m,m);
for i = 1 : m
    for j = 1 : m
        gauKer(i,j) = exp(-.5 * (norm(A(i,:) - A(j,:),2))^2);
    end
end

%% fit the kernelized least square
alpha = inv(gauKer + lambda * eye(size(gauKer))) * b;


%% compute classification accuracy
prediction = nan(m,1);
for i = 1 : m
    temp = 0;
    xnew = A(i,:);
    for j = 1 : m
        temp = temp + alpha(j) * exp(-.5 * (norm(xnew - A(j,:),2) )^2);
    end
    prediction(i) = sign(temp);
end

sum(prediction)
accuracy = sum(prediction == b) / length(b)

%% plot the prediction 
figure; 
hold on;
for i=1:m
    a = A(i,:);
    if prediction(i)==1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('Ridge regression with Guanssian kernel', 'fontsize',14)