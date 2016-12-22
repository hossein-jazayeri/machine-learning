function p = predict(theta, X)

m = size(X, 1);

p = zeros(m, 1);

z = X * theta;
h = sigmoid(z);
p = p + 0.5;
p = h >= p;

end
