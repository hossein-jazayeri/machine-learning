function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

h = a3;

[v, p] = max(h, [], 2);

end
