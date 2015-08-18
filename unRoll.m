function [Theta1, Theta2, Theta3] = unRoll(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels)
% Reshape nn_params back into the parameters Theta1, Theta2, and Theta3 the weight matrices
% for our 3 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
             
theta2Str = 1 + hidden_layer_size*(1 + input_layer_size);
theta2End = (theta2Str - 1) + hidden_layer_size * (hidden_layer_size + 1);
Theta2 = reshape(nn_params(theta2Str : theta2End), ...
                 hidden_layer_size, hidden_layer_size + 1);  

Theta3 = reshape(nn_params(theta2End + 1:end), ...
                 num_labels, (hidden_layer_size + 1));
end

