function weights = cnn_l2_train_lbfgs(xtrain, ytrain, params, xval, yval)


% -- initialization
weights = cnn_l2_init(params);

addpath(genpath('utils/minFunc_2012/'));
theta = cnn_l2_roll(weights);


% lbfgs
options.method = 'lbfgs';
options.maxiter = params.maxiter;


opttheta = minFunc(@(p) cnn_l2_grad_roll(p, xtrain, ytrain, params, xval, yval), theta, options);
weights = cnn_l2_unroll(opttheta, params);


% -- filename to save
fname_mat = sprintf('results/%s.mat', params.fname);
save(fname_mat, 'weights', 'params');


return;