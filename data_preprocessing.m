function [x, label, xorig, yorig] = data_preprocessing(maxsize)

if ~exist('maxsize', 'var'),
    maxsize = inf;
end

load psoasGSarray.mat gsArray;

xorig = cell(length(gsArray), 1);
yorig = cell(length(gsArray), 1);

rsize = zeros(length(gsArray), 1);
csize = zeros(length(gsArray), 1);

for i = 1:length(gsArray),
    img = gsArray(i).inputImage;
    [r, c] = size(img);
    rsize(i) = r;
    csize(i) = c;
end

ratio = rsize./csize;
mean_rsize = min(maxsize/14, round(mean(rsize)/14))*14;
mean_csize = round(mean_rsize./mean(ratio)/14)*14;


x = zeros(mean_rsize, mean_csize, length(gsArray));
label = zeros(mean_rsize, mean_csize, length(gsArray));

for i = 1:length(gsArray),
    img = gsArray(i).inputImage;
    mask = gsArray(i).outputMask;
    xorig{i} = img;
    yorig{i} = mask;
    
    img = imresize(img, [mean_rsize, mean_csize], 'bicubic');
    mask = imresize(mask, [mean_rsize, mean_csize], 'bicubic');
    img = im2double(img);
    
    x(:, :, i) = img;
    label(:, :, i) = mask;
end

return;