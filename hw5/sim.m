sigma = 2;
n = 10000;
x = randn(n,1);
f = (1/(2*sigma)) * exp(- (abs(x) / sigma));
mean(f)
mle = sum(abs(x))/n