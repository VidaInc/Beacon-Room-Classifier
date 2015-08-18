function p = predict(Theta1, Theta2, Theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

a1 = [ones(m, 1) X];
z1 = sigmoid(Theta1*a1');
a2 = [ones(1, size(z1, 2)) ; z1];
z2 = sigmoid(Theta2*a2);
a3 = [ones(1, size(z2, 2)) ; z2];
hyp = sigmoid(a3'*Theta3');
[~,p] = max(hyp, [], 2);

% =========================================================================


end
