clear ; close all; clc

fprintf('Plotting Data ...\n')
data = load('data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); % number of training examples

plotData(X, y);

fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:, 1)];

% initialize fitting parameters
theta = zeros(2, 1);

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

computeCost(X, y, theta);

theta = gradientDescent(X, y, theta, alpha, iterations);

fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off;

predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1 * 10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2 * 10000);

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
  	t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
  end
end


% Because of the way meshgrids work in the surf command, we need to transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20));
xlabel('\theta_0');
ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
