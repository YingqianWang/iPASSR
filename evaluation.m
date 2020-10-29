function evaluation
clear all;
clc;
close all;

Method = 'iPASSR';                      % Method for evaluation
factor = 2;                                  % Upsampling factor
stereo_boundary = 64;       % Cropping left 64 pixels in the left view for evaluation
image_boundary = 0;         % Cropping image boundaries for evaluation
Datasets = {'Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury'};
GT_folder = './data/test/';
ResultsPath = ['./results/', Method, '_', num2str(factor), 'xSR/'];
dataset_num = length(Datasets);

for DatasetIndex = 1 : dataset_num
    DatasetName = Datasets{DatasetIndex};
    GT_DataFolder = [GT_folder, DatasetName, '/hr/'];
    GTfiles = dir(GT_DataFolder);
    GTfiles(1:2) = [];
    sceneNum = length(GTfiles);
    RT_DataFolder = dir([ResultsPath, DatasetName, '/*.png']);
    
    txtName = [ResultsPath, Method, '_', num2str(factor), 'xSR_', DatasetName, '.txt'];
    fp = fopen(txtName, 'w+');
    fclose(fp);
    
    for iScene = 1 : sceneNum
        scene_name = GTfiles(iScene).name;
        
        fprintf('Running Scene_%s in Dataset %s......\n',scene_name, DatasetName);        
        gt_left = imread([GT_DataFolder, GTfiles(iScene).name, '/hr0.png']);
        gt_right = imread([GT_DataFolder, GTfiles(iScene).name, '/hr1.png']);
        
        sr_left = imread([ResultsPath, DatasetName, '/', scene_name, '_L.png']);
        sr_right = imread([ResultsPath, DatasetName, '/', scene_name, '_R.png']);
        
        gt_left_crop = gt_left(:, 1+stereo_boundary : end, :);
        sr_left_crop = sr_left(:, 1+stereo_boundary : end, :);
        
        [psnr_left_crop(iScene), ssim_left_crop(iScene)] = cal_metrics(sr_left_crop, gt_left_crop, image_boundary);
        
        [psnr_left, ssim_left] = cal_metrics(sr_left, gt_left, image_boundary);
        [psnr_right, ssim_right] = cal_metrics(sr_right, gt_right, image_boundary);
        psnr_stereo(iScene) = (psnr_left + psnr_right) / 2;
        ssim_stereo(iScene) = (ssim_left + ssim_right) / 2;
        fp = fopen(txtName, 'a');
        fprintf(fp, '\n %03d \t %4f \t %4f \t %4f \t %4f \t \n',...
            iScene, psnr_left_crop(iScene), ssim_left_crop(iScene), psnr_stereo(iScene), ssim_stereo(iScene));
        fclose(fp);
        
    end
    psnr_left_crop_avg = mean(psnr_left_crop); psnr_stereo_avg = mean(psnr_stereo);
    ssim_left_crop_avg = mean(ssim_left_crop); ssim_stereo_avg = mean(ssim_stereo);
    fp = fopen(txtName, 'a');
    fprintf(fp, '\n %s\t %4f \t %4f \t %4f \t %4f \t %4f \t %4f \t \n',...
        'AVG', psnr_left_crop_avg, ssim_left_crop_avg, psnr_stereo_avg, ssim_stereo_avg);
    fclose(fp);
    
    psnr_left_crop = []; psnr_stereo = [];
    ssim_left_crop = []; ssim_stereo = [];
end
end


function [psnr, ssim] = cal_metrics(im0, im1, boundary)
im0_ = im2double(im0(1+boundary : end-boundary, 1+boundary : end-boundary, :));
im1_ = im2double(im1(1+boundary : end-boundary, 1+boundary : end-boundary, :));
temp = (im0_ - im1_).^2;
mse = mean(temp(:));
psnr = 10*log10(1 / mse);
ssim = cal_ssim(im0_, im1_);
end

function ssim = cal_ssim( im1, im2)
[~, ~, ch] = size(im1);
ssim = 0;
if (ch == 1)
    ssim = ssim_index( im1, im2);
else
    for i = 1:ch
        ssim = ssim + ssim_index(im1, im2);
    end
    ssim = ssim/3;
end
end

function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)
if (nargin < 2 || nargin > 5)
    mssim = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    mssim = -Inf;
    ssim_map = -Inf;
    return;
end

[M N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        mssim = -Inf;
        ssim_map = -Inf;
        return
    end
    %window = fspecial('gaussian', 11, 1.5);	%
    window = fspecial('average', 7);
    K(1) = 0.01;					% default settings
    K(2) = 0.03;					%
    L = 1;                                     %
end

img1 = double(img1);
img2 = double(img2);

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end
mssim = mean2(ssim_map);
end