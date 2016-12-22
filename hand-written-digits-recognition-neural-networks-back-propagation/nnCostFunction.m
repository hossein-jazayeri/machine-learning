function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, lambda)

ThetaSize = hidden_layer_size * (input_layer_size + 1);

Theta1 = reshape(nn_params(1:ThetaSize), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + ThetaSize):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

y_matrix = eye(num_labels)(y,:);

cost_unregularized = sum(sum((-y_matrix .* log(h)) - ((1 - y_matrix) .* log(1 - h)))) / m;

regularize_J = (lambda / (2 * m)) * ( sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)) );

J = cost_unregularized + regularize_J;

D1 = D2 = 0;

for i = 1:m
  a1 = X(i,:);
  a1 = [1; a1'];

  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  delta3 = a3 - y_matrix(i,:)';
  delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
  delta2 = delta2(2:end);

  D2 += delta3 * a2';
  D1 += delta2 * a1';
endfor

Theta1_grad = D1 / m + ((lambda / m) * Theta1);
Theta2_grad = D2 / m + ((lambda / m) * Theta2);

Theta1_grad(:,1) = D1(:,1) / m;
Theta2_grad(:,1) = D2(:,1) / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
