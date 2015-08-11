function theta = trainClassifier(data)

y = data(:, 13);

% Add Polynomial Features
X = horzcat(ones(size(data(:,1))), mapFeature(data(:, 1), data(:, 2), data(:, 3), 5), ...
    mapFeature(data(:, 4), data(:, 5), data(:, 6), 2),data(: , 7:12));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 3000);

% Optimize
[theta, ~, ~] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

end

