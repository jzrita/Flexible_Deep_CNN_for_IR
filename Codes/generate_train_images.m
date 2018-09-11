close all;
clear
clc
addpath('...');
%% settings
stride = 21;
JPEG_Quality=10;
depth_network=20;

if JPEG_Quality==10
lamda_lq=0.04;%for q20
%lamda_lq1=0.01;%for q20
elseif JPEG_Quality==20
lamda_lq=0.02;%for q10
%lamda_lq1=0.03;%for q10
elseif JPEG_Quality==30
lamda_lq=0.01;
else
    lamda_lq=0.005;
end
%====================
lamda_gt=0.02;

if depth_network==10
size_input = 31; %original 21
size_label = 31;
elseif depth_network==20
size_input = 31;
size_label = 31;    
end

%% geneate BSDS500 training data
folder_GT='BSDS500_400\GT';
filepaths_GT_rgb = dir(fullfile(folder_GT,'*.jpg'));

savepath_name_1 = ['STCNN_train_q',num2str(JPEG_Quality),'_001_whole_1.h5'];
savepath_name_2 = ['STCNN_train_q',num2str(JPEG_Quality),'_001_whole_2.h5'];
savepath_name_3 = ['STCNN_train_q',num2str(JPEG_Quality),'_001_whole_3.h5'];
savepath_name_4 = ['STCNN_train_q',num2str(JPEG_Quality),'_001_whole_4.h5'];

order1 = randperm(length(filepaths_GT_rgb));

%% initialization
data_t = zeros(size_input, size_input, 1, 1);
label_t = zeros(size_label, size_label, 1, 1);
data_s = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;
for i = 1 : length(filepaths_GT_rgb)/4
    im_gt_rgb = imread(fullfile(folder_GT,filepaths_GT_rgb(order1(i)).name));
    im_gt_ycbcr = rgb2ycbcr(im_gt_rgb);
    imwrite(im_gt_ycbcr(:,:,1),'test.jpg','jpg','Quality',JPEG_Quality);
    im_gt_y=im2double(im_gt_ycbcr(:,:,1));
    im_lq_y=im2double(imread('test.jpg'));
    
    %% structure_texture separation
    [im_lq_text, im_lq_struct] =  TV_L2_Decomp(im_lq_y, lamda_lq) ; %ROF_decomp(y , 0.05, 100, 1) ;
    [im_gt_text, ~] =  TV_L2_Decomp(im_gt_y, lamda_gt) ;
    [hei,wid] = size(im_gt_y);    
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1           
            subim_input_lq_t = im_lq_text(x : x+size_input-1, y : y+size_input-1);
            subim_input_lq_s = im_lq_struct(x : x+size_input-1, y : y+size_input-1);           
            subim_label = im_gt_y(x : x+size_label-1, y : y+size_label-1);
            subim_label_t = im_gt_text(x : x+size_label-1, y : y+size_label-1);
            count=count+1;
            data_t(:, :, 1, count) = subim_input_lq_t;
            label_t(:, :, 1, count) = subim_label_t;
            data_s(:, :, 1, count) = subim_input_lq_s;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data_T = data_t(:, :, 1, order);
label_T = label_t(:, :, 1, order);
data_S = data_s(:, :, 1, order);
label_GT = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata_t = data_T(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_t = label_T(:,:,1,last_read+1:last_read+chunksz);
    batchdata_s = data_S(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label_GT(:,:,1,last_read+1:last_read+chunksz);
    
    startloc = struct('dat_t',[1,1,1,totalct+1], 'lab_t', [1,1,1,totalct+1],'dat_s',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_multi_data(savepath_name_1, batchdata_t, batchlabs_t, batchdata_s,batchlabs,~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_name_1);

%% initialization
data_t = zeros(size_input, size_input, 1, 1);
label_t = zeros(size_label, size_label, 1, 1);
data_s = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;

for i = 1+length(filepaths_GT_rgb)/4 : length(filepaths_GT_rgb)/2
    
    im_gt_rgb = imread(fullfile(folder_GT,filepaths_GT_rgb(order1(i)).name));
    im_gt_ycbcr = rgb2ycbcr(im_gt_rgb);
    imwrite(im_gt_ycbcr(:,:,1),'test.jpg','jpg','Quality',JPEG_Quality);
    im_gt_y=im2double(im_gt_ycbcr(:,:,1));
    im_lq_y=im2double(imread('test.jpg'));
    
    %% structure_texture separation
    [im_lq_text, im_lq_struct] =  TV_L2_Decomp(im_lq_y, lamda_lq) ; %ROF_decomp(y , 0.05, 100, 1) ;
    [im_gt_text, ~] =  TV_L2_Decomp(im_gt_y, lamda_gt) ;
    [hei,wid] = size(im_gt_y);    
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1           
            subim_input_lq_t = im_lq_text(x : x+size_input-1, y : y+size_input-1);
            subim_input_lq_s = im_lq_struct(x : x+size_input-1, y : y+size_input-1);           
            subim_label = im_gt_y(x : x+size_label-1, y : y+size_label-1);
            subim_label_t = im_gt_text(x : x+size_label-1, y : y+size_label-1);
            count=count+1;
            data_t(:, :, 1, count) = subim_input_lq_t;
            label_t(:, :, 1, count) = subim_label_t;
            data_s(:, :, 1, count) = subim_input_lq_s;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data_T = data_t(:, :, 1, order);
label_T = label_t(:, :, 1, order);
data_S = data_s(:, :, 1, order);
label_GT = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata_t = data_T(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_t = label_T(:,:,1,last_read+1:last_read+chunksz);
    batchdata_s = data_S(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label_GT(:,:,1,last_read+1:last_read+chunksz);
    
    startloc = struct('dat_t',[1,1,1,totalct+1], 'lab_t', [1,1,1,totalct+1],'dat_s',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_multi_data(savepath_name_2, batchdata_t, batchlabs_t, batchdata_s,batchlabs,~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_name_2);

%% initialization
data_t = zeros(size_input, size_input, 1, 1);
label_t = zeros(size_label, size_label, 1, 1);
data_s = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;
for i = (1+length(filepaths_GT_rgb)/2):(length(filepaths_GT_rgb)-length(filepaths_GT_rgb)/4)
    im_gt_rgb = imread(fullfile(folder_GT,filepaths_GT_rgb(order1(i)).name));
    im_gt_ycbcr = rgb2ycbcr(im_gt_rgb);
    imwrite(im_gt_ycbcr(:,:,1),'test.jpg','jpg','Quality',JPEG_Quality);
    im_gt_y=im2double(im_gt_ycbcr(:,:,1));
    im_lq_y=im2double(imread('test.jpg'));
    
    %% structure_texture separation
   [im_lq_text, im_lq_struct] =  TV_L2_Decomp(im_lq_y, lamda_lq) ; %ROF_decomp(y , 0.05, 100, 1) ;
    [im_lq_text1, im_lq_struct1] =  TV_L2_Decomp(im_lq_y, lamda_lq1) ;    
    im_lq_text_diff=im_lq_text-im_lq_text1;
    [im_gt_text, im_gt_struct] =  TV_L2_Decomp(im_gt_y, lamda_gt) ;
    [hei,wid] = size(im_gt_y);    
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1           
            subim_input_lq_t = im_lq_text_diff(x : x+size_input-1, y : y+size_input-1);
            subim_input_lq_s = im_lq_struct(x : x+size_input-1, y : y+size_input-1);           
            subim_label = im_gt_y(x : x+size_label-1, y : y+size_label-1);
            subim_label_t = im_gt_text(x : x+size_label-1, y : y+size_label-1);
            count=count+1;
            data_t(:, :, 1, count) = subim_input_lq_t;
            label_t(:, :, 1, count) = subim_label_t;
            data_s(:, :, 1, count) = subim_input_lq_s;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data_T = data_t(:, :, 1, order);
label_T = label_t(:, :, 1, order);
data_S = data_s(:, :, 1, order);
label_GT = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata_t = data_T(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_t = label_T(:,:,1,last_read+1:last_read+chunksz);
    batchdata_s = data_S(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label_GT(:,:,1,last_read+1:last_read+chunksz);
    
    startloc = struct('dat_t',[1,1,1,totalct+1], 'lab_t', [1,1,1,totalct+1],'dat_s',[1,1,1,totalct+1],'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_multi_data(savepath_name_3, batchdata_t, batchlabs_t, batchdata_s,batchlabs,~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_name_3);

%% initialization
data_t = zeros(size_input, size_input, 1, 1);
label_t = zeros(size_label, size_label, 1, 1);
data_s = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;
for i = 1+(length(filepaths_GT_rgb)-length(filepaths_GT_rgb)/4) : length(filepaths_GT_rgb)
    im_gt_rgb = imread(fullfile(folder_GT,filepaths_GT_rgb(order1(i)).name));
    im_gt_ycbcr = rgb2ycbcr(im_gt_rgb);
    imwrite(im_gt_ycbcr(:,:,1),'test.jpg','jpg','Quality',JPEG_Quality);
    im_gt_y=im2double(im_gt_ycbcr(:,:,1));
    im_lq_y=im2double(imread('test.jpg'));
    
    %% structure_texture separation
   [im_lq_text, im_lq_struct] =  TV_L2_Decomp(im_lq_y, lamda_lq) ; %ROF_decomp(y , 0.05, 100, 1) ;
    [im_lq_text1, im_lq_struct1] =  TV_L2_Decomp(im_lq_y, lamda_lq1) ;    
    im_lq_text_diff=im_lq_text-im_lq_text1;
    [im_gt_text, im_gt_struct] =  TV_L2_Decomp(im_gt_y, lamda_gt) ;
    [hei,wid] = size(im_gt_y);    
    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1           
            subim_input_lq_t = im_lq_text_diff(x : x+size_input-1, y : y+size_input-1);
            subim_input_lq_s = im_lq_struct(x : x+size_input-1, y : y+size_input-1);           
            subim_label = im_gt_y(x : x+size_label-1, y : y+size_label-1);
            subim_label_t = im_gt_text(x : x+size_label-1, y : y+size_label-1);
            count=count+1;
            data_t(:, :, 1, count) = subim_input_lq_t;
            label_t(:, :, 1, count) = subim_label_t;
            data_s(:, :, 1, count) = subim_input_lq_s;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data_T = data_t(:, :, 1, order);
label_T = label_t(:, :, 1, order);
data_S = data_s(:, :, 1, order);
label_GT = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata_t = data_T(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_t = label_T(:,:,1,last_read+1:last_read+chunksz);
    batchdata_s = data_S(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label_GT(:,:,1,last_read+1:last_read+chunksz);
    
    startloc = struct('dat_t',[1,1,1,totalct+1], 'lab_t', [1,1,1,totalct+1],'dat_s',[1,1,1,totalct+1],'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_multi_data(savepath_name_4, batchdata_t, batchlabs_t, batchdata_s,batchlabs,~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_name_4);
delete('test.jpg');