% LS geometry
% use the "rotate" tool on the Matlab plot
% to visualize the span of A cols (magenta) and
% orthogonality of the residual (dashed)
% b vector in blue, bhat = A*xhat in green

clear
close all

% define A and b
A = [1 1 1; 1 -1 1]';
b = [1 1 0]';


% plot columns in A and b
hold on
a1 = [0 0 0;A(1,1) A(2,1) A(3,1)];
t=plot3(a1(:,1),a1(:,2),a1(:,3),'m');
set(t,'linewidth',3)
a2 = [0 0 0;A(1,2) A(2,2) A(3,2)];
t=plot3(a2(:,1),a2(:,2),a2(:,3),'m');
set(t,'linewidth',3)
bv = [0 0 0;b(1) b(2) b(3)];
t=plot3(bv(:,1),bv(:,2),bv(:,3),'b');
set(t,'linewidth',3)
axis([-1 2 -1 2 -1 2])
axis square
grid
pause

% generate and plot 50 points that are random
% linear combinations of the columns in A
n=5000;
X = A*randn(2,n);
scatter3(X(1,:),X(2,:),X(3,:),'k.')
pause

% computer LS solution to min ||b-Ax||
xhat = inv(A'*A)*A'*b;
bhat = A*xhat;

r = [bhat(1) bhat(2) bhat(3);b(1) b(2) b(3)];
t=plot3(r(:,1),r(:,2),r(:,3),'k:','linewidth',.1);
set(t,'linewidth',3)
scatter3(bhat(1),bhat(2),bhat(3),'bo')
pause

% plot A*xhat
bhatv = [0 0 0;bhat(1) bhat(2) bhat(3)];
t=plot3(bhatv(:,1),bhatv(:,2),bhatv(:,3),'g');
set(t,'linewidth',3)

