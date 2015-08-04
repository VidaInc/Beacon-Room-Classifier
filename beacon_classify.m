% Initialization
clear ; close all; clc

% Load Data
load('data');
X = data(:, 1:10); y = data(:, 11);
%X = horzcat(ones(size(X, 1), 1), data(:,1:10));

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = horzcat(mapFeature(data(:, 1), data(:, 2), data(:, 3)), data(: , 4:10));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


