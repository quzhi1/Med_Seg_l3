function [rec, prec, ap] = cnn_l2_evaluate(x, y, weights, params)

pen = cnn_l2_infer(x, weights, params);
yhat = cnn_l2_recon(pen, weights, params);

[rec, prec, ap] = compute_ap(yhat(:), y(:));

return;