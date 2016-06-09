clear all; clf; 
%% no intercept model
n = 100; 
A = 1 + 2 * randn(n,1);
y = A + 2 * randn(n,1);

% compute OLS solution 
x = (A'*A)\ A'* y;

hold on 
% plot the data 
scatter(A,y)
% plot the regression line 
range = min(A):max(A);
line1 = x*range;
plot(range,line1)


%% add a column of 1
X = [ones(n,1) A];
x2 = (X'*X)\ X'* y;
line2 = x2(1) + x2(2)*range;
plot(range,line2)
hold off

legend('data','without intercept','with intercept')
