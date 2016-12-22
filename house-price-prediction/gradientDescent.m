function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  h = X * theta;
  theta(1, 1) = theta(1, 1) - (alpha / m) * sum(h - y);
  theta(2, 1) = theta(2, 1) - (alpha / m) * sum((h - y) .* X(:,2));
  
  cost = computeCost(X, y, theta);
  J_history(iter) = cost;
end

end
