function [cost, grad] = cnn_l2_grad(weights, xtrain, ytrain, params)


batchsize = size(xtrain, 4);

% -- feed-forward inference
[pen, hid_bu] = cnn_l2_infer(xtrain, weights, params);
[yhat, hid_td] = cnn_l2_recon(pen, weights, params);


% -- compute cost
cost = cross_entropy(ytrain, yhat)/batchsize;

grad = replicate_struct(weights, 0);

% -- backprop
dobj = (yhat - ytrain)/batchsize;


% objective -> hid_td
grad.visbias = grad.visbias + sum(dobj, 4);
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        grad.hidvis(:,:,c,b) = grad.hidvis(:,:,c,b) + convn(dobj(:,:,b,:), hid_td(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end
end

% dh (hid_td)
dh = zeros(size(hid_td));
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        dh(:,:,c,:) = dh(:,:,c,:) + convn(dobj(:,:,b,:), weights.hidvis(end:-1:1,end:-1:1,c,b), 'valid');
    end
end
switch params.nonlinearity,
    case 'relu',
        dobj = dh.*(hid_td > 0);
    case 'sigmoid',
        dobj = dh.*hid_td.*(1-hid_td);
    case 'linear',
        dobj = dh;
end


% hid_td -> pen
grad.hidbias_td = grad.hidbias_td + sum(sum(sum(dobj, 1), 2), 4);
for b = 1:size(weights.penhid, 4),
    for c = 1:size(weights.penhid, 3),
        grad.penhid(:,:,c,b) = grad.penhid(:,:,c,b) + convn(dobj(:,:,b,:), pen(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end
end

% dp (pen)
dp = zeros(size(pen));
for b = 1:size(weights.penhid, 4),
    for c = 1:size(weights.penhid, 3),
        dp(:,:,c,:) = dp(:,:,c,:) + convn(dobj(:,:,b,:), weights.penhid(end:-1:1,end:-1:1,c,b), 'valid');
    end
end
switch params.nonlinearity,
    case 'relu',
        dobj = dp.*(pen > 0);
    case 'sigmoid',
        dobj = dp.*pen.*(1-pen);
    case 'linear',
        dobj = dp;
end


% pen -> hid_bu
grad.penbias = grad.penbias + sum(sum(sum(dobj, 1), 2), 4);
for b = 1:size(weights.hidpen, 4),
    for c = 1:size(weights.hidpen, 3),
        grad.hidpen(:,:,c,b) = grad.hidpen(:,:,c,b) + convn(hid_bu(:,:,c,:), dobj(end:-1:1,end:-1:1,b,end:-1:1), 'valid');
    end
end

% dh (hid_bu)
dh = zeros(size(hid_bu));
for b = 1:size(weights.hidpen, 4),
    for c = 1:size(weights.hidpen, 3),
        dh(:,:,c,:) = dh(:,:,c,:) + convn(weights.hidpen(:,:,c,b), dobj(:,:,b,:), 'full');
    end
end
switch params.nonlinearity,
    case 'relu',
        dobj = dh.*(hid_bu > 0);
    case 'sigmoid',
        dobj = dh.*hid_bu.*(1-hid_bu);
    case 'linear',
        dobj = dh;
end


% hid_bu -> input
grad.hidbias_bu = grad.hidbias_bu + sum(sum(sum(dobj, 1), 2), 4);
for b = 1:size(weights.vishid, 4),
    for c = 1:size(weights.vishid, 3),
        grad.vishid(:,:,c,b) = grad.vishid(:,:,c,b) + convn(xtrain(:,:,c,:), dobj(end:-1:1,end:-1:1,b,end:-1:1), 'valid');
    end
end


return;