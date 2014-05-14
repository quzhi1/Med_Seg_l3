function demo_seg(opt, numhid, ws, maxiter, optvis, optmask, alpha)


if ~exist('maxsize', 'var'),
    % resize images into fixed size while keeping aspect ratio
    % maxsize = min(rowsize, colsize)
    maxsize = 70;
end
if ~exist('opt', 'var'),
    % optimization method, 'sgd' or 'lbfgs'
    opt = 'lbfgs';
end
if ~exist('numhid', 'var'),
    % number of convolutional filters
    numhid = 50;
end
if ~exist('ws', 'var'),
    % filter size
    ws = 16;
end
if ~exist('maxiter', 'var'),
    % maximum # of iteration
    maxiter = 300;
end
if ~exist('optvis', 'var'),
    optvis = 1;
end
if ~exist('optmask', 'var'),
    % post processing with mask
    optmask = 1;
end
if ~exist('alpha', 'var'),
    % masking hyperparameter
    % larger the alpha, mask gets close to step function
    alpha = 0.1;
end

if ~optmask,
    alpha = 0;
end
if alpha == 0,
    optmask = 0;
end


% ---------------------------------------
% load data
% ---------------------------------------

[x, label, xorig, yorig] = data_preprocessing(maxsize);
x = permute(x, [1 2 4 3]);
label = permute(label, [1 2 4 3]);

dim = size(x);

tr_id = 1:120;
val_id = 121:150;
ts_id = 151:size(x, 4);

xtrain = x(:, :, :, tr_id);
xval = x(:, :, :, val_id);
xtest = x(:, :, :, ts_id);
ytrain = label(:, :, :, tr_id);
yval = label(:, :, :, val_id);
ytest = label(:, :, :, ts_id);

xtest_orig = xorig(ts_id);
yval_orig = yorig(val_id);
ytest_orig = yorig(ts_id);

if optmask,
    mask_prior = mean(ytrain, 4);
    mask_prior = max(0, tanh(mask_prior*alpha));
end


% ---------------------------------------
% compute chance acc and mean AP
% (when predicted all 0)
% ---------------------------------------

% val
chance_ap_val = zeros(size(xval, 4), 1);
chance_acc_val = zeros(size(xval, 4), 1);
for i = 1:size(xval, 4),
    yc = yval(:, :, :, i);
    [~, ~, ap] = compute_ap(0*yc(:), yc(:));
    chance_ap_val(i) = ap;
    chance_acc_val(i) = 1-mean(yc(:));
end

% test
chance_ap_test = zeros(size(xtest, 4), 1);
chance_acc_test = zeros(size(xtest, 4), 1);
for i = 1:size(xtest, 4),
    yc = ytest(:, :, :, i);
    [~, ~, ap] = compute_ap(0*yc(:), yc(:));
    chance_ap_test(i) = ap;
    chance_acc_test(i) = 1-mean(yc(:));
end

fprintf('chance AP: val = %g (std %g), test = %g (std %g)\n', mean(chance_ap_val), std(chance_ap_val), mean(chance_ap_test), std(chance_ap_test));
fprintf('chance ACC: val = %g (std %g), test = %g (std %g)\n', mean(chance_acc_val), std(chance_acc_val), mean(chance_acc_test), std(chance_acc_test));


% ---------------------------------------
% train convolutional neural network
% ---------------------------------------

% -- preprocessing (subtract mean and divide by variance)
m = mean(mean(xtrain, 1), 2);
stds = std(reshape(xtrain, size(xtrain, 1)*size(xtrain, 2), size(xtrain, 3)*size(xtrain, 4)));
stds = permute(reshape(stds, size(xtrain, 3), size(xtrain, 4)), [3 4 1 2]);
xtrain = bsxfun(@rdivide, bsxfun(@minus, xtrain, m), stds);

m = mean(mean(xval, 1), 2);
stds = std(reshape(xval, size(xval, 1)*size(xval, 2), size(xval, 3)*size(xval, 4)));
stds = permute(reshape(stds, size(xval, 3), size(xval, 4)), [3 4 1 2]);
xval = bsxfun(@rdivide, bsxfun(@minus, xval, m), stds);

m = mean(mean(xtest, 1), 2);
stds = std(reshape(xtest, size(xtest, 1)*size(xtest, 2), size(xtest, 3)*size(xtest, 4)));
stds = permute(reshape(stds, size(xtest, 3), size(xtest, 4)), [3 4 1 2]);
xtest = bsxfun(@rdivide, bsxfun(@minus, xtest, m), stds);


% -- train convNet
params = struct(...
    'dataset', 'seg', ...
    'numch', dim(3), ...
    'numhid', numhid, ...
    'numout', 1, ...
    'optimize', opt, ...
    'ws', ws, ...
    'rs', dim(1), ...
    'cs', dim(2), ...
    'nonlinearity', 'relu', ...
    'eps', 0.0001, ...
    'eps_decay', 0.01, ...
    'maxiter', maxiter, ...
    'batchsize', 10, ...
    'momentum_change', 0, ...
    'momentum_init', 0.33, ...
    'momentum_final', 0.5, ...
    'verbose', 0);

disp(params);

if strcmp(params.optimize, 'sgd'),
    params.fname = sprintf('%s_%s_rs_%d_cs_%d_ws_%d_nh_%d_%s_eps_%g_itr_%d_bs_%d', ...
        params.dataset, params.optimize, params.rs, params.cs, params.ws, params.numhid, params.nonlinearity, ...
        params.eps, params.maxiter, params.batchsize);
    
    weights = cnn_train(xtrain, ytrain, params, xval, yval);
elseif strcmp(params.optimize, 'lbfgs'),
    params.fname = sprintf('%s_%s_rs_%d_cs_%d_ws_%d_nh_%d_%s_itr_%d', ...
        params.dataset, params.optimize, params.rs, params.cs, params.ws, params.numhid, params.nonlinearity, params.maxiter);
    
    if exist(sprintf('results/%s.mat', params.fname), 'file'),
        load(sprintf('results/%s.mat', params.fname), 'weights', 'params');
    else
        weights = cnn_train_lbfgs(xtrain, ytrain, params, xval, yval);
    end
end


[~, ~, ap_val] = cnn_evaluate(xval, yval, weights, params);
[~, ~, ap_test] = cnn_evaluate(xtest, ytest, weights, params);


fprintf('AP: val = %g, test = %g (%s)\n', ap_val, ap_test, params.fname);
fid = fopen(sprintf('log/seg_%s.txt', params.optimize), 'a+');
fprintf(fid, 'AP: val = %g, test = %g (%s)\n', ap_val, ap_test, params.fname);
fclose(fid);


% ----------------------------------------------
% learn additional classifier with mask prior
% ----------------------------------------------

if optmask,
    if exist(sprintf('results/%s_al%g.mat', params.fname, alpha), 'file'),
        load(sprintf('results/%s_al%g.mat', params.fname, alpha), 'w', 'b', 'mask_prior');
    else
        yhtrain = zeros(size(ytrain));
        
        for i = 1:size(xtrain, 4),
            xc = xtrain(:, :, :, i);
            
            % inference
            h = cnn_infer(xc, weights, params);
            yhat = cnn_recon(h, weights, params);
            yhtrain(:, :, :, i) = yhat.*mask_prior;
        end
        
        yhtrain = yhtrain(:);
        ytrain_flat = ytrain(:);
        
        [w, b] = logistic_regression(yhtrain', ytrain_flat', 0);
        
        clear yhtrain ytrain_flat;
        save(sprintf('results/%s_al%g.mat', params.fname, alpha), 'w', 'b', 'mask_prior');
    end
end


% compute again
ap_val = zeros(size(xval, 4), 1);
acc_val = zeros(size(xval, 4), 1);
ap_val_orig = zeros(size(xval, 4), 1);
acc_val_orig = zeros(size(xval, 4), 1);
for i = 1:size(xval, 4),
    xc = xval(:, :, :, i);
    yc = yval(:, :, :, i);
    yc_orig = yval_orig{i};
    
    % inference
    h = cnn_infer(xc, weights, params);
    yhat = cnn_recon(h, weights, params);
    if optmask,
        yhat = sigmoid(w*yhat.*mask_prior + b);
    end
    
    yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
    
    [~, ~, ap] = compute_ap(yhat(:), yc(:));
    ap_val(i) = ap;
    acc_val(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    
    [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
    ap_val_orig(i) = ap;
    acc_val_orig(i) = 1-mean(yc_orig(:) ~= (yhat_orig(:) > 0.5));
end


ap_test = zeros(size(xtest, 4), 1);
acc_test = zeros(size(xtest, 4), 1);
ap_test_orig = zeros(size(xtest, 4), 1);
acc_test_orig = zeros(size(xtest, 4), 1);

te = 0;
for i = 1:size(xtest, 4),
    xc = xtest(:, :, :, i);
    yc = ytest(:, :, :, i);
    yc_orig = ytest_orig{i};
    
    % inference
    ts = tic;
    h = cnn_infer(xc, weights, params);
    yhat = cnn_recon(h, weights, params);
    te = te + toc(ts);
    
    if optmask,
        yhat = sigmoid(w*yhat.*mask_prior + b);
    end
    yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
    
    [~, ~, ap] = compute_ap(yhat(:), yc(:));
    ap_test(i) = ap;
    acc_test(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    
    [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
    ap_test_orig(i) = ap;
    acc_test_orig(i) = 1-mean(yc_orig(:) ~= (yhat_orig(:) > 0.5));
end
inf_time = te/size(xtest, 4);


fprintf('AP: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(ap_val), std(ap_val), mean(ap_test), std(ap_test), alpha, inf_time, params.fname);
fprintf('ACC: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(acc_val), std(acc_val), mean(acc_test), std(acc_test), alpha, inf_time, params.fname);
fprintf('AP (orig): val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(ap_val_orig), std(ap_val_orig), mean(ap_test_orig), std(ap_test_orig), alpha, inf_time, params.fname);
fprintf('ACC (orig): val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(acc_val_orig), std(acc_val_orig), mean(acc_test_orig), std(acc_test_orig), alpha, inf_time, params.fname);


fid = fopen(sprintf('log/avg_seg_%s.txt', params.optimize), 'a+');
fprintf(fid, 'AP: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(ap_val), std(ap_val), mean(ap_test), std(ap_test), alpha, inf_time, params.fname);
fprintf(fid, 'ACC: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(acc_val), std(acc_val), mean(acc_test), std(acc_test), alpha, inf_time, params.fname);
fprintf(fid, 'AP (orig): val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(ap_val_orig), std(ap_val_orig), mean(ap_test_orig), std(ap_test_orig), alpha, inf_time, params.fname);
fprintf(fid, 'ACC (orig): val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
    mean(acc_val_orig), std(acc_val_orig), mean(acc_test_orig), std(acc_test_orig), alpha, inf_time, params.fname);
fprintf(fid, '\n');
fclose(fid);




% -- visualization
if optvis,
    % filter visualization
    vishid = reshape(weights.vishid, params.ws^2*params.numch, params.numhid);
    fig = figure(1);
    display_network_nonsquare(vishid);
    print(sprintf('%s/%s_filter.png', 'vis', params.fname), fig, '-dpng');
    
    % visible bias
    visbias = weights.visbias;
    fig = figure(1);
    imagesc(visbias); colormap gray; axis off;
    print(sprintf('%s/%s_visbias.png', 'vis', params.fname), fig, '-dpng');
    
    
    % best 5 prediction from test set
    [~, id] = sort(ap_test_orig, 'descend');
    
    for i = 1:5,
        xc = xtest(:, :, :, id(i));
        xc_orig = xtest_orig{id(i)};
        yc_orig = ytest_orig{id(i)};
        
        % inference
        h = cnn_infer(xc, weights, params);
        yhat = cnn_recon(h, weights, params);
        yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
        [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
        
        fig = figure(1);
        set(fig, 'Position', [100, 100, 768, 1408]);
        subplot(2,2,1); imagesc(xc_orig); colormap gray; axis off;
        title('original image');
        subplot(2,2,2); imagesc(yc_orig); colormap gray; axis off;
        title('ground truth label');
        subplot(2,2,3); imagesc(yhat_orig); colormap gray; axis off;
        title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, 150+id(i)));
        
        if optmask,
            yhat = sigmoid(w*yhat.*mask_prior + b);
            yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
            [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
            
            subplot(2,2,4); imagesc(yhat_orig); colormap gray; axis off;
            title(sprintf('predicted label (mask), mean AP = %g, studyid = %d', ap, 150+id(i)));
        end
        
        print(sprintf('%s/%s_prediction_studyid_%d_al_%g.png', 'vis', params.fname, 150+id(i), alpha), fig, '-dpng');
        close(fig);
    end
    
    
    % worst 5 prediction from test set
    for i = length(id):-1:length(id)-4,
        xc = xtest(:, :, :, id(i));
        xc_orig = xtest_orig{id(i)};
        yc_orig = ytest_orig{id(i)};
        
        % inference
        h = cnn_infer(xc, weights, params);
        yhat = cnn_recon(h, weights, params);
        yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
        [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
        
        fig = figure(1);
        set(fig, 'Position', [100, 100, 768, 1408]);
        subplot(2,2,1); imagesc(xc_orig); colormap gray; axis off;
        title('original image');
        subplot(2,2,2); imagesc(yc_orig); colormap gray; axis off;
        title('ground truth label');
        subplot(2,2,3); imagesc(yhat_orig); colormap gray; axis off;
        title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, 150+id(i)));
        
        if optmask,
            yhat = sigmoid(w*yhat.*mask_prior + b);
            yhat_orig = imresize(yhat, size(yc_orig), 'bicubic');
            [~, ~, ap] = compute_ap(yhat_orig(:), yc_orig(:));
            
            subplot(2,2,4); imagesc(yhat_orig); colormap gray; axis off;
            title(sprintf('predicted label (mask), mean AP = %g, studyid = %d', ap, 150+id(i)));
        end
        
        print(sprintf('%s/%s_prediction_studyid_%d_al_%g.png', 'vis', params.fname, 150+id(i), alpha), fig, '-dpng');
        close(fig);
    end
end


return;
