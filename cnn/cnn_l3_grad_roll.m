function [cost, grad] = cnn_l3_grad_roll(theta, xtrain, ytrain, params, xval, yval)


% -- unroll
weights = cnn_l3_unroll(theta, params);


% -- cost and gradient
[cost, grad] = cnn_l3_grad(weights, xtrain, ytrain, params);


% -- roll
grad = cnn_l3_roll(grad);


% -- evaluate
if exist('xval', 'var'),
    [~, ~, ap] = cnn_l3_evaluate(xval, yval, weights, params);
    fprintf('val AP = %g\n', ap);
end


return;
