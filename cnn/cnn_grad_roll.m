function [cost, grad] = cnn_grad_roll(theta, xtrain, ytrain, params, xval, yval)


% -- unroll
weights = cnn_unroll(theta, params);


% -- cost and gradient
[cost, grad] = cnn_grad(weights, xtrain, ytrain, params);


% -- roll
grad = cnn_roll(grad);


% -- evaluate
if exist('xval', 'var'),
    [~, ~, ap] = cnn_evaluate(xval, yval, weights, params);
    fprintf('val AP = %g\n', ap);
end


return;