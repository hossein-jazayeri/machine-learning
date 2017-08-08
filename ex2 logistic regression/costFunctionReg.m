function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

z = X * theta;
h = sigmoid(z);
theta2 = theta(2:end);
J = sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) / m + (lambda / (2 * m)) * sum(theta2 .^ 2);

grad = sum((h - y) .* X) / m + (lambda / m) * theta';
grad2 = sum((h - y) .* X) / m;
grad(1) = grad2(1);

end
