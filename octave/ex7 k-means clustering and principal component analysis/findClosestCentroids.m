function idx = findClosestCentroids(X, centroids)
%   FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);

idx = zeros(size(X,1), 1);

% Go over every example, find its closest centroid, and store
% the index inside idx at the appropriate location.
% Concretely, idx(i) should contain the index of the centroid
% closest to example i. Hence, it should be a value in the 
% range 1..K

m = size(X, 1);
for i=1:m
    min_error = inf;
    for k=1:K
        error = norm(X(i,:) - centroids(k,:));
        if error < min_error
            min_error = error;
            idx(i) = k;
        end
    end
end

end
