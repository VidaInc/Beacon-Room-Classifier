clear ; close all; clc

hidden_layer_size = 500;
num_labels = 4;         

% Load Data
load('data');

% Add Polynomial Features
X = horzcat(mapFeature(data(:, 1), data(:, 2), data(:, 3), 3), data(:, 4:12));
y = data(:, 13);

m = size(X, 1);
input_layer_size  = size(X, 2);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; ... 
    initial_Theta3(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 7500);

lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

[Theta1, Theta2, Theta3] = unRoll(nn_params, input_layer_size, ... 
    hidden_layer_size, num_labels);
             
pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);