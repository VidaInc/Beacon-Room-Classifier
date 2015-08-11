function out = mapFeature(X1, X2, X3, degree)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2, X3) maps three input features
%   to quadratic features used in the regularization.
%
%   Returns a new feature array with more features
%
%   Inputs X1, X2 X3 must be the same size

out = [];
for i = 1:degree
    for j = 0:i
        for k = 0:j 
            out(:, end+1) = (X1.^(i-j)).*(X2.^(j-k)).*(X3.^k);
        end
    end
end

end