addpath cnn/;
addpath utils/;

if ~exist('results', 'dir'),
    mkdir('results');
end
if ~exist('log', 'dir'),
    mkdir('log');
end
if ~exist('vis', 'dir'),
    mkdir('vis');
end



