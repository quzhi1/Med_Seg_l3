function weights = cnn_l2_unroll(theta, params)

idx = 0;

weights.vishid = reshape(theta(idx+1:idx+params.ws^2*params.numch*params.numhid), ...
    params.ws, params.ws, params.numch, params.numhid);
idx = idx + numel(weights.vishid);

weights.hidbias_bu = permute(theta(idx+1:idx+params.numhid), [2 3 1]);
idx = idx + numel(weights.hidbias_bu);

weights.hidpen = reshape(theta(idx+1:idx+params.ws2^2*params.numhid*params.numpen), ...
    params.ws2, params.ws2, params.numhid, params.numpen);
idx = idx + numel(weights.hidpen);

weights.penbias = permute(theta(idx+1:idx+params.numpen), [2 3 1]);
idx = idx + numel(weights.penbias);

weights.penhid = reshape(theta(idx+1:idx+params.ws2^2*params.numpen*params.numhid), ...
    params.ws2, params.ws2, params.numpen, params.numhid);
idx = idx + numel(weights.penhid);

weights.hidbias_td = permute(theta(idx+1:idx+params.numhid), [2 3 1]);
idx = idx + numel(weights.hidbias_td);

weights.hidvis = reshape(theta(idx+1:idx+params.ws^2*params.numhid*params.numout), ...
    params.ws, params.ws, params.numhid, params.numout);
idx = idx + numel(weights.hidvis);

weights.visbias = reshape(theta(idx+1:idx+params.rs*params.cs*params.numout), params.rs, params.cs, params.numout);
idx = idx + numel(weights.visbias);

assert(idx == length(theta));


return;