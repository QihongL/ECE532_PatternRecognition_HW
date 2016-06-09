%% set up the matrix and vectors
clear all; clc;
A = [3 4 1; 2 3 2]';
x1 = [1; 10; 100];
x2 = [3;4];

%% 1. cost of making one widget and one gizmo
% linear combination of rows
result1 = zeros(1,size(A,2));
for i = 1 : length(x1)
    % multiply coefficient in x with row of A
    for j = 1 : length(A(i,:))
        result1(j) = result1(j) + x1(i) * A(i,j);
    end
end
% check my answer
if any(result1 ~= x1' * A)
    warning ('wrong!')
end


%% materials, parts and labors needed for making 3 widgets and 4 gizmos
% consider it as a linear combination of columns of A
result2 = zeros(size(A,1),size(x2,2));
for j = 1:length(x2)
    for i = 1 : length(A(:,j))
        result2(i) = result2(i) + A(i,j) * x2(j);
    end
end
% check my answer
if any(result2 ~= A * x2)
    warning ('wrong!')
end


%% total costs of making 3 widgets and 4 gizmos
result3 = 0;
for i = 1:size(x2)
    result3 = result3 + result1(i) * x2(i);
end
% check my answer
if any(result3 ~= x1' * A * x2)
    warning ('wrong!')
end
