% =====================================
% CNN feedforward inference
% =====================================

function h = cnn_infer(x, weights, params)


batchsize = size(x, 4);

vishidlr = zeros(params.ws, params.ws, params.numhid, params.numch);
for c = 1:params.numch,
    vishidlr(:,:,:,c) = reshape(weights.vishid(end:-1:1, end:-1:1, c, :),[params.ws,params.ws,params.numhid]);
end

h = repmat(weights.hidbias, [size(x,1)-params.ws+1, size(x,2)-params.ws+1, 1, batchsize]);
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


return;