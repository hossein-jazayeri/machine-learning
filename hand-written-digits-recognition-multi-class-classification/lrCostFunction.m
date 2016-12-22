function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);

z = X * theta;
h = sigmoid(z);
unregularized_cost = sum((-y .* log(h)) - ((1 - y) .* log(1 - h))) / m;
J = unregularized_cost + (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

grad = sum((h - y) .* X) / m + (lambda / m) * theta';
grad2 = sum((h - y) .* X) / m;
grad(1) = grad2(1);

grad = grad(:);

end
