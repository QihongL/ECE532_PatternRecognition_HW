clear all; clc
load fisheriris.mat

% 3d data 
X = meas(:,1:3);
scatter3(X(:,1),X(:,2),X(:,3))

[U,S,V] = svd(X);



pcs = V(:,1:2);
projMat = pcs*inv(pcs'*pcs)*pcs';
projectedX = X * projMat;

pcs2 = pca(X)
pcs2 = pcs2(:,1:2)
projMat2 = pcs2*inv(pcs2'*pcs2)*pcs2';
Xp2 = X * projMat2;

figure;
scatter3(Xp2(:,1),Xp2(:,2),Xp2(:,3))