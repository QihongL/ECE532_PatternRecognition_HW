%% Homework 2 - Question 6 - emotion recognition
clear all; clc;
load('face_emotion_data.mat')

%% Problam a
beta = inv(X' * X) * X' * y

%% Problem b
% if a new face has input x, which is a column 
% then check if x' * beta is closer to 1 or -1 

%% Problam c
% they are using the same unit so don't normalize it 
abs(beta)
sort(abs(beta))
