clear ; close all; clc

input_layer_size = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10

fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
