function [J, grad] = costFunction(theta, X, y)

m = length(y);

z = X * theta;
h = sigmoid(z);
J = sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) / m;

grad = sum((h - y) .* X) / m;

end
