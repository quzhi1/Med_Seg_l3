% =====================================
% CNN feedforward inference
% =====================================

function [hy, p, h] = cnn_l3_infer(x, weights, params)


batchsize = size(x, 4);

% l1
vishidlr = zeros(params.ws, params.ws, params.numhid, params.numch);
for c = 1:params.numch,
    vishidlr(:,:,:,c) = reshape(weights.vishid(end:-1:1, end:-1:1, c, :), [params.ws,params.ws,params.numhid]);
end

h = repmat(weights.hidbias_bu, [size(x,1)-params.ws+1, size(x,2)-params.ws+1, 1, batchsize]);
for c = 1:params.numch,
    for d = 1:params.numhid,
        h(:,:,d,:) = h(:,:,d,:) + convn(x(:,:,c,:), vishidlr(:,:,d,c), 'valid');
    end
end

switch params.nonlinearity,
    case 'relu',
        h = max(0, h);
    case 'sigmoid',
        h = sigmoid(h);
end


% l2
hidpenlr = zeros(params.ws2, params.ws2, params.numpen, params.numhid);
for c = 1:params.numhid,
    hidpenlr(:,:,:,c) = reshape(weights.hidpen(end:-1:1, end:-1:1, c, :), [params.ws2,params.ws2,params.numpen]);
end

p = repmat(weights.penbias_bu, [size(h,1)-params.ws2+1, size(h,2)-params.ws2+1, 1, batchsize]);
for c = 1:params.numhid,
    for d = 1:params.numpen,
        p(:,:,d,:) = p(:,:,d,:) + convn(h(:,:,c,:), hidpenlr(:,:,d,c), 'valid');
    end
end

switch params.nonlinearity,
    case 'relu',
        p = max(0, p);
    case 'sigmoid',
        p = sigmoid(p);
end

%l3
penhyperlr = zeros(params.ws3, params.ws3, params.numhyper, params.numpen);
for c = 1:params.numpen,
    penhyperlr(:,:,:,c) = reshape(weights.penhyper(end:-1:1, end:-1:1, c, :), [params.ws3,params.ws3,params.numhyper]);
end

hy = repmat(weights.hyperbias, [size(p,1)-params.ws3+1, size(p,2)-params.ws3+1, 1, batchsize]);
for c = 1:params.numpen,
    for d = 1:params.numhyper,
        hy(:,:,d,:) = hy(:,:,d,:) + convn(p(:,:,c,:), penhyperlr(:,:,d,c), 'valid');
    end
end

switch params.nonlinearity,
    case 'relu',
        hy = max(0, hy);
    case 'sigmoid',
        hy = sigmoid(hy);
end

return;
