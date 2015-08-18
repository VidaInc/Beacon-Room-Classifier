function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1, Theta2, and Theta3 the weight matrices
% for our 3 layer neural network
[Theta1, Theta2, Theta3] = unRoll(nn_params, input_layer_size, ... 
    hidden_layer_size, num_labels);

% Setup some useful variables
m = size(X, 1);

eyesOfLabels = eye(num_labels);
y = eyesOfLabels(y,:);
 
a1 = [ones(m, 1) X];
 
z2 = a1 * Theta1';
a2 = sigmoid(z2);
 
n2 = size(a2, 1);
a2 = [ones(n2,1) a2];
 
z3 = a2 * Theta2';
a3 = sigmoid(z3);

n3 = size(a3, 1);
a3 = [ones(n3,1) a3];

z4 = a3 * Theta3';
a4 = sigmoid(z4);

regularization = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end)).^2)) ... 
    + sum(sum((Theta2(:,2:end)).^2)) + sum(sum((Theta3(:,2:end)).^2)));
 
J = ((1/m) * sum(sum((-y .* log(a4))-((1-y) .* log(1-a4))))) + regularization;
 
 
delta_4 = a4 - y;
delta_3 = (delta_4 * Theta3(:,2:end)) .* sigmoidGradient(z3);
delta_2 = (delta_3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
 
 
delta_cap3 = delta_4' * a3; 
delta_cap2 = delta_3' * a2;
delta_cap1 = delta_2' * a1;
 
Theta1_grad = ((1/m) * delta_cap1) + ((lambda/m) * (Theta1));
Theta2_grad = ((1/m) * delta_cap2) + ((lambda/m) * (Theta2));
Theta3_grad = ((1/m) * delta_cap3) + ((lambda/m) * (Theta3));
 
Theta1_grad(:,1) = Theta1_grad(:,1) - ((lambda/m) * (Theta1(:,1)));
Theta2_grad(:,1) = Theta2_grad(:,1) - ((lambda/m) * (Theta2(:,1)));
Theta3_grad(:,1) = Theta3_grad(:,1) - ((lambda/m) * (Theta3(:,1)));
% =========================================================================
 
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
