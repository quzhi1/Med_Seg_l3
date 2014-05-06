function weights = cnn_train_lbfgs(xtrain, ytrain, params, xval, yval)


% -- initialization
weights = cnn_init(params);

addpath(genpath('utils/minFunc_2012/'));
theta = cnn_roll(weights);


% lbfgs
options.method = 'lbfgs';
options.maxiter = params.maxiter;


opttheta = minFunc(@(p) cnn_grad_roll(p, xtrain, ytrain, params, xval, yval), theta, options);
weights = cnn_unroll(opttheta, params);


% -- filename to save
fname_mat = sprintf('results/%s.mat', params.fname);
save(fname_mat, 'weights', 'params');


return;