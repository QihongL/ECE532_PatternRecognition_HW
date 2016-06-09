function [X, y, btrue] = generateBluringData(k,sigma, plot)
% deblurring
% k = 30;
% sigma = 0.01;

n = 500;
% generate random piecewise constant signal
x = zeros(n,1);
x(1) = randn;
for i=2:n
    if (rand < .95)
        x(i) = x(i-1);
    else
        x(i) = randn;
    end
end

% generate k-point averaging function
h = ones(1,k)/k;

% make A matrix for blurring
m = n+k-1;
for i=1:m
    if i<=k
        A(i,1:i) = h(1:i);
    else
        A(i,i-k+1:i) = h;
    end
end
A = A(:,1:n);

% blurred signal + noise
b = A*x+sigma*randn(m,1);

% plot
if plot
    figure(1)
    subplot(211)
    plot(x)
    t=title('signal')
    set(gca,'Fontsize',16)
    set(t,'Fontsize',16)
    subplot(212)
    plot(b(1:n))
    axis('tight')
    t=title('blurred and noisy version')
    set(t,'Fontsize',16)
    set(gca,'Fontsize',16)
end

%% save it
X = A;
y = b;
btrue = x;
% save('blurringData.mat','X','y', 'btrue')
end
