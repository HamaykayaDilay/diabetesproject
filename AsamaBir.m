clear all, close all, clc

%cd 'C:\Users\hamay\mtlb' in MATLAB online you don't need this so it's
%optional

%% Loading the data

load diabetes
X=diabetes(:,1:8);
Y=diabetes(:,9);

%summary(categorical(Y))
% 0: Normal
% 1: Sick
% TP 11
% TN 00

% rescale, normalize, min-max

%% Decision Tree
% doc fitctree
odevDT(X,Y)
% 'MaxNumSplits', 3, 'MinLeafSize',22

%% SVM
%doc fitcsvm
odevSVM(X,Y)
% 'KernelFunction','linear', 'RBF', 'gaussian'
% 'BoxConstraint', 1

%% KNN
%doc fitcknn
odevKNN(X,Y)
%'NumNeighbors',10
% 'DistanceWeight', 'Equal'


%% Discriminant Analysis
% doc fitcdiscr
odevDisAnalysis(X,Y)
% 'Gamma', 0.2
