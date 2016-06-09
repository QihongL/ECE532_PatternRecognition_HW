clear; close all;clc
training_data
% set lambda
lambda = 1e-5;

%% compute the polynomial kernal
m = size(A,1);
ker = nan(m,m);
for i=1:m
    for j=1:m
        ker(i,j) = (dot(A(i,:), A(j,:)) +1 )^2;
    end
end

% %% test kernel 
% phi = [A(:,1).^2 sqrt(2).*A(:,1).*A(:,2) A(:,2).^2 sqrt(2)*A(:,1) sqrt(2)*A(:,2) ones(m,1)];
% kert = phi * phi';
% [ker(:,1) kert(:,1)]

%% fit the kernelized least square
alpha = inv(ker + lambda * eye(size(ker))) * b;


%% compute classification accuracy
prediction = nan(m,1);
predict_t = nan(m,1);
% compute the kernel LS prediction 1 by 1
for i = 1 : m
    temp = 0; 
    xnew = A(i,:);
    % compute the kernel LS prediction for 1 example
    for j = 1 : m
        temp = temp + alpha(j) * (dot(xnew, A(j,:)) +1 )^2;
    end
    prediction(i) = sign(temp);
end

accuracy = sum(prediction == b) / length(b)

%% plot the prediction 
figure; 
hold on;
for i = 1 : m
    a = A(i,:);
    if prediction(i) == 1
        plot(a(1),a(2),'b.');
    else
        plot(a(1),a(2),'r.');
    end
end
axis('square')
title('Ridge regression with polynomial kernel', 'fontsize',14)