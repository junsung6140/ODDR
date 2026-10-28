function measure_shadow(shadowdir, freedir, maskdir)
% Standard shadow removal evaluation (PSNR / SSIM / RMSE in shadow, non-shadow and all regions).
% All images are resized to 256x256. "RMSE" is the mean absolute error in LAB color space,
% following the convention of prior shadow removal work. Table 1 and Table 2 of the paper
% report the PSNR / SSIM lines and the PP-Lab line (all = per-image mean, regions = per-pixel).
%
% Usage (MATLAB, file names must be sorted in the same order in the three folders):
%   measure_shadow('results/AISTD/', 'data/AISTD/test/test_C_fixed_official/', 'data/AISTD/test/test_B/')
%
% Adapted from ShadowFormer (https://github.com/GuoLanqing/ShadowFormer/blob/main/evaluation/measure_shadow.m,
% MIT License).

if shadowdir(end) ~= '/', shadowdir = [shadowdir '/']; end
if freedir(end) ~= '/', freedir = [freedir '/']; end
if maskdir(end) ~= '/', maskdir = [maskdir '/']; end
SD = dir([shadowdir '*.png']);
FD = dir([freedir '*.png']);
MD = dir([maskdir '*.png']);

total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(SD,1)); 
smae=zeros(1,size(SD,1)); 
nmae=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    sname = strcat(shadowdir,SD(i).name);
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir, MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    
    
    
    f = double(f)/255;
    s = double(s)/255;
    
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);
%     

    nmask=~m;       % mask of the non-shadow region
    smask=~nmask;   % mask of the shadow region
%     avg_s = mean(mean(mean(s)));
%     avg_f = mean(mean(mean(f)));
%     s = s.*avg_f ./avg_s;
    ppsnr(i)=psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)= ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% MAE, per pixel
    allmae(i)=sum(dist(:))/size(f,1)/size(f,2);
    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  

    disp(i);
end
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allmae),total_distn/total_pixeln,total_dists/total_pixels);
end
