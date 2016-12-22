function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(m, 1);

X = [ones(m, 1) X];

h = sigmoid(X * all_theta');
[v, p] = max(h, [], 2);

end
