function [rec, prec, ap] = compute_ap(score, gt)

gt = gt*2 - 1; % +1 and -1

[~, si]=sort(-score);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/(sum(gt>0)+0.0001);
prec=tp./(fp+tp+0.0001);

ap = compute_ap_sub(rec(:), prec(:));

% plot(rec, prec, '-');

return;

function ap = compute_ap_sub(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

return;