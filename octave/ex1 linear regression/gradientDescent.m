function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters    
    h = X * theta;
    theta(1, 1) = theta(1, 1) - (alpha / m) * sum(h - y);
    theta(2, 1) = theta(2, 1) - (alpha / m) * sum((h - y) .* X(:,2));
    

    % Save the cost J in every iteration    
    cost = computeCost(X, y, theta);
    J_history(iter) = cost;

end

end
