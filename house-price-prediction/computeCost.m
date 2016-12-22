function J = computeCost(X, y, theta)

m = length(y);
predictions = X * theta;
sqauredErrors = (predictions - y) .^ 2;
J = (1 / (2 * m)) * sum(sqauredErrors);

end
