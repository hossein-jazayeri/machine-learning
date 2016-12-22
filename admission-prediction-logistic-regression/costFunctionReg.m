function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);

z = X * theta;
h = sigmoid(z);
theta2 = theta(2:end);
J = sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) / m + (lambda / (2 * m)) * sum(theta2 .^ 2);

grad = sum((h - y) .* X) / m + (lambda / m) * theta';
grad2 = sum((h - y) .* X) / m;
grad(1) = grad2(1);

end
