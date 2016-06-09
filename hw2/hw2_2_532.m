%% Homework 2 - Question 2
clear all; clc;

%% a
A = [1 1; 1 -1; 1 1];
b = [1 1 0]';

x = (A' * A) \ A' * b

%% b - can I do this plotting in matlab? 