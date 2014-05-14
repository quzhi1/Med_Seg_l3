params.ws = 3;
params.ws2 = 2;
params.ws3 = 2;
params.numch = 2;
params.numhid = 6;
params.numpen = 4;
params.numhyper = 2;
params.numout = 1;
params.rs = 10;
params.cs = 10;
params.nonlinearity = 'sigmoid';
params.stdinit = 0.1;

batchsize = 10;

xtrain = 0.3*randn(params.rs, params.cs, params.numch, batchsize);
ytrain = double(rand(params.rs, params.cs, params.numout, batchsize) < 0.5);


weights = cnn_l3_init(params);
theta = cnn_l3_roll(weights);

addpath ~/scratch/kihyuks/library/GradCheck/;
[diff, ngrad, tgrad] = GradCheck(@(p) cnn_l2_grad_roll(p, xtrain, ytrain, params), theta);

ngrad = cnn_l3_unroll(ngrad, params);
tgrad = cnn_l3_unroll(tgrad, params);

fnames = fieldnames(ngrad);

for i = 1:length(fnames),
    nA = getfield(ngrad, fnames{i});
    tA = getfield(tgrad, fnames{i});
    
    fprintf('%s diff = %g\n', fnames{i}, norm(nA(:)-tA(:))/norm(nA(:)+tA(:)));
end
