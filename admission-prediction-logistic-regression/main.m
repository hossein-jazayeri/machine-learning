clear ; close all; clc

data = load('data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

fprintf(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);

plotData(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

plotDecisionBoundary(theta, X, y);

hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission probability of %f\n\n'], prob);

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
