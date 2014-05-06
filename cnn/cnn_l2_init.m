function weights = cnn_l2_init(params)

ws = params.ws;
ws2 = params.ws2;
rs = params.rs;
cs = params.cs;
numch = params.numch;
numhid = params.numhid;
numpen = params.numpen;
numout = params.numout;
if isfield(params, 'stdinit'),
    stdinit = params.stdinit;
else
    stdinit = 0.01;
end

weights = struct;

weights.vishid = stdinit*randn(ws, ws, numch, numhid);
weights.hidbias_bu = zeros(1, 1, numhid);
weights.hidpen = stdinit*randn(ws2, ws2, numhid, numpen);
weights.penbias = zeros(1, 1, numpen);
weights.penhid = stdinit*randn(ws2, ws2, numpen, numhid);
weights.hidbias_td = zeros(1, 1, numhid);
weights.hidvis = stdinit*randn(ws, ws, numhid, numout);
weights.visbias = zeros(rs, cs, numout);

return;
