%% Machine Learning Exercise 6 | Support Vector Machines

clear ; close all; clc

%% =============== Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data1 to X, y
load('data1.mat');

plotData(X, y);

fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1];
x2 = [0 4 -1];
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f: %f' ...
         '\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 2 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data2 to X, y
load('data2.mat');

plotData(X, y);

fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% SVM Parameters
C = 1;
sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster.
% However, in practice, you will want to run the training to convergence.

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 3 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data3 to X, y
load('data3.mat');

plotData(X, y);

[C, sigma] = dataset3Params(X, y, Xval, yval);

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;
