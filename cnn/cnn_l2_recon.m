function [y, h] = cnn_l2_recon(p, weights, params)

batchsize = size(p, 4);

% pen -> hid
penhid = weights.penhid;
h = repmat(weights.hidbias_td, [size(p,1)+params.ws2-1, size(p,2)+params.ws2-1, 1, batchsize]);
for b = 1:params.numpen,
    for c = 1:params.numhid,
        h(:,:,c,:) = h(:,:,c,:) + convn(p(:,:,b,:), penhid(:,:,b,c), 'full');
    end
end

switch params.nonlinearity,
    case 'relu',
        h = max(0, h);
    case 'sigmoid',
        h = sigmoid(h);
end

% hid -> vis
hidvis = weights.hidvis;
vbiasmat = repmat(weights.visbias, [1 1 1 batchsize]);

y = vbiasmat;
for b = 1:params.numhid,
    for c = 1:params.numout,
        y(:,:,c,:) = y(:,:,c,:) + convn(h(:,:,b,:), hidvis(:,:,b,c), 'full');
    end
end

y = sigmoid(y);

return;