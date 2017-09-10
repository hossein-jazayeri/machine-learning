function [J, grad] = costFunction(theta, X, y)
%   COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

z = X * theta;
h = sigmoid(z);
J = sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) / m;

grad = sum((h - y) .* X) / m;

end
