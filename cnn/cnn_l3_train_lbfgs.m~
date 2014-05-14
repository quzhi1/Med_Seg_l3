function weights = cnn_l3_train_lbfgs(xtrain, ytrain, params, xval, yval)

sprintf('start training with 3 layer machine.\n');

% -- initialization
weights = cnn_l3_init(params);

addpath(genpath('utils/minFunc_2012/'));
theta = cnn_l3_roll(weights);


% lbfgs
options.method = 'lbfgs';
options.maxiter = params.maxiter;


opttheta = minFunc(@(p) cnn_l3_grad_roll(p, xtrain, ytrain, params, xval, yval), theta, options);
weights = cnn_l3_unroll(opttheta, params);


% -- filename to save
fname_mat = sprintf('results/%s.mat', params.fname);
save(fname_mat, 'weights', 'params');


return;
