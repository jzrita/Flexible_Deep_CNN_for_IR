clear
folder0='...';
datapath='...';
if(~isdir(datapath))
    mkdir(datapath);
end
filepaths=dir(fullfile(folder0,'*.bmp'));
Nimg=length(filepaths);
%% Rotate images
for i=1:Nimg
    filename=filepaths(i).name;
    [imaddress,imname,type]=fileparts(filepaths(i).name);
    image=imread(fullfile(folder0,filename));
    im1=rot90(image,1);
    im2=rot90(image,2);
    im3=rot90(image,3);
    imwrite(im1,[datapath imname, '_rot90' '.bmp']);
    imwrite(im2,[datapath imname, '_rot180' '.bmp']);
    imwrite(im3,[datapath imname, '_rot270' '.bmp']);
end
% mirror images
filepaths=dir(fullfile(folder0,'*.bmp'));
Nimg=length(filepaths);

for i=1:Nimg
    filename=filepaths(i).name;
    [imaddress,imname,type]=fileparts(filepaths(i).name);
    image=imread(fullfile(folder0,filename));
    im1=fliplr(image);%horizontal flip
    imwrite(im1,[datapath imname, '_h' '.bmp']);
end