function weights = cnn_init(params)

ws = params.ws;
rs = params.rs;
cs = params.cs;
numch = params.numch;
numhid = params.numhid;
numout = params.numout;

weights = struct;

weights.vishid = 0.01*randn(ws, ws, numch, numhid);
weights.hidbias = zeros(1, 1, numhid);
weights.hidvis = 0.01*randn(ws, ws, numhid, numout);
weights.visbias = zeros(rs, cs, numout);

return;
