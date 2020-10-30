function GenerateTrainingPatches
clear all;
clc;
close all;

Datasets = {'Middlebury', 'Flickr1024'};  % Original datasets for training data generation
scale = 2;       % down-scaling factor

idx_patch = 1;
for idx_dataset = 1 : length(Datasets)
    dataset = Datasets{idx_dataset};
    img_list = dir(['./', dataset, '/*.png']);
    
    for idx_file = 1 : 2 : length(img_list)                
        img_0 = imread(['./', dataset, '/', img_list(idx_file).name]);
        img_1 = imread(['./', dataset, '/', img_list(idx_file + 1).name]);
        
        %% generate HR & LR images
        if strcmp(dataset, 'Middlebury') == 1
            img_0 = imresize(img_0, 1/2, 'bicubic');
            img_1 = imresize(img_1, 1/2, 'bicubic');
        end
        img_hr_0 = modcrop(img_0, scale);
        img_hr_1 = modcrop(img_1, scale);
        img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
        img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
        
        %% extract patches of size 30*90 with stride 20
        
        for x_lr = 3 : 20 : size(img_lr_0,1) - 33
            for y_lr = 3 : 20 : size(img_lr_0,2) - 93
                x_hr = (x_lr - 1) * scale + 1;
                y_hr = (y_lr - 1) * scale + 1;
                hr_patch_0 = img_hr_0(x_hr : (x_lr+29)*scale, y_hr : (y_lr+89)*scale, :);
                hr_patch_1 = img_hr_1(x_hr : (x_lr+29)*scale, y_hr : (y_lr+89)*scale, :);
                lr_patch_0 = img_lr_0(x_lr : x_lr+29, y_lr : y_lr+89, :);
                lr_patch_1 = img_lr_1(x_lr : x_lr+29, y_lr : y_lr+89, :);
                
                mkdir(['./patches_x', num2str(scale), '/', num2str(idx_patch, '%06d')]);
                imwrite(hr_patch_0, ['./patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/hr0.png']);
                imwrite(hr_patch_1, ['./patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/hr1.png']);
                imwrite(lr_patch_0, ['./patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/lr0.png']);
                imwrite(lr_patch_1, ['./patches_x', num2str(scale), '/', num2str(idx_patch, '%06d'), '/lr1.png']);
                fprintf([num2str(idx_patch, '%06d'), ' training samples have been generated...\n']);
                idx_patch = idx_patch + 1;
            end
        end
    end
end
end


function img_cropped = modcrop(img, scale_factor)
h = size(img, 1);
w = size(img, 2);

img_cropped = img(1:floor(h/scale_factor)*scale_factor, 1:floor(w/scale_factor)*scale_factor, :);
end