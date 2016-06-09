%% load the data matrix
clear all; clc; close all; 
load('jesterdata.mat')
%% compute the SVD
[U,S,V] = svd(X, 'econ');
%% dimensionality reduction 
subspaceDim = 3; 
% project rows
pcsV = V(:,1:subspaceDim);
projMatV = pcsV*inv(pcsV'*pcsV)*pcsV';
projectedX_v = X * projMatV;
XpV = projectedX_v(:,1:subspaceDim);
% project columns
pcsU = U(:,1:subspaceDim);
projMatU = pcsU*inv(pcsU'*pcsU)*pcsU';
projectedX_u = X' * projMatU;
XpU = projectedX_u(:,1:subspaceDim);

%% plot rows 
subplot(3,2,1)
scatter3(XpV(:,1),XpV(:,2),XpV(:,3))
title('Projected data (Rows) onto the PC space', 'fontsize', 14)
subplot(3,2,3)
scatter3(XpV(:,1),XpV(:,2),XpV(:,3))
title('Rotated plot1', 'fontsize', 14)
subplot(3,2,5)
scatter3(XpV(:,1),XpV(:,2),XpV(:,3))
title('Rotated plot2', 'fontsize', 14)

%% plot columns 
subplot(3,2,2)
scatter3(XpU(:,1),XpU(:,2),XpU(:,3), '.')
title('Projected data (Columns) onto the PC space', 'fontsize', 14)
subplot(3,2,4)
scatter3(XpU(:,1),XpU(:,2),XpU(:,3), '.')
title('Rotated plot1', 'fontsize', 14)
subplot(3,2,6)
scatter3(XpU(:,1),XpU(:,2),XpU(:,3), '.')