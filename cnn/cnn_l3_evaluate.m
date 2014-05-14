function [rec, prec, ap] = cnn_l3_evaluate(x, y, weights, params)

sprintf('Evaluate AP of this iteratiron.\n');

[hyper, ~, ~] = cnn_l3_infer(x, weights, params);
yhat = cnn_l3_recon(hyper, weights, params);

[rec, prec, ap] = compute_ap(yhat(:), y(:));

return;