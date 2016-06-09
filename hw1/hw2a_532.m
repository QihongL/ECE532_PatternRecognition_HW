clear all; 
% C = AB express as a sum of rank 1 matrices
% m=3;
% k=3;
% n=3;
% 
% A = randn(m,k);
% B = randn(k,n);
% 
% C = zeros(m,n);
% for i = 1 : k
%     rank(A(:,i) * B(i,:));
%     C = C + A(:,i) * B(i,:);
% end


%% ECE 532 HW1 - 2A
clear all; 
% express X'X as the sum of rank1 matrices 
p = 10; 
n = 3;

X = randn(p,n);
XT = X';
sum = zeros(n,n); % X'X will have dim(n,n)
for i = 1 : n 
    rank(XT(:,i) * X(i,:))
    sum = sum + XT(:,i) * X(i,:);
end

answer = XT * X;
