function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

a1 = [ones(m, 1) X];
a2 = sigmoid(Theta1*a1');
a2 = [ones(1, size(a2, 2)) ; a2];
hyp = sigmoid(a2'*Theta2');
[~,p] = max(hyp, [], 2);

% =========================================================================


end
