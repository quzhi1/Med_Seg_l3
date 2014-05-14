params.ws = 3;
params.numch = 2;
params.numhid = 4;
params.numout = 1;
params.rs = 10;
params.cs = 10;
params.nonlinearity = 'sigmoid';

batchsize = 10;

xtrain = 0.1*randn(params.rs, params.cs, params.numch, batchsize);
ytrain = double(rand(params.rs, params.cs, params.numout, batchsize) < 0.5);

weights = cnn_init(params);
theta = cnn_roll(weights);

addpath ~/scratch/kihyuks/library/GradCheck/;
[diff, ngrad, tgrad] = GradCheck(@(p) cnn_grad_roll(p, xtrain, ytrain, params), theta);

ngrad = cnn_unroll(ngrad, params);
tgrad = cnn_unroll(tgrad, params);

fnames = fieldnames(ngrad);

for i = 1:length(fnames),
    nA = getfield(ngrad, fnames{i});
    tA = getfield(tgrad, fnames{i});
    
    fprintf('%s diff = %g\n', fnames{i}, norm(nA(:)-tA(:))/norm(nA(:)+tA(:)));
end