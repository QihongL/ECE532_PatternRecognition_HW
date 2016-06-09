%% Homework 2 - Question 4 - cereal calorie prediction problem 
clear all; clc; 

% %% Problam a
% A = [25 0 1; 20 1 2; 40 1 6];
% b = [110 110 210]';
% 
% x = inv(A' * A) * A' * b
% 
% %% Problam b
% x_true = [4 9 4]';
% 
% A * x_true

%% Problem c 
A = [25 15 10 0 1; 20 12 8 1 2; 40 30 10 1 6; 30 15 15 0 3; 35 20 15 2 4];
b = [104 97 193 132 174]';
ATA = A' * A
ATb = A' * b 
rref([ATA, ATb])
rref([A b])

c = randi(999);
xg = [4-c c c 9 4]';
xt = [4 0 0 9 4]';
ATA * xg
A * xg
