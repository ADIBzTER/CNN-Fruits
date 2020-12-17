%%
% This scripts standardize the .jpg images
Dir = 'Fruits\apples\'; % write the name of your sub- folder containg images for a specific label
imagefiles = dir(fullfile(Dir,'*.jpg'));
for i = 1 : length(imagefiles)
filename = strcat(Dir ,imagefiles(i).name);
sf=imread(filename);
sf=imresize(sf, [100 100]); % standard size of the images, up to the user
newName = sprintf('%01d.jpg', i); % new file name, up to user
imwrite(sf,newName); % the new images will be saved in the current folder
end

%%
datasetPath = fullfile('Fruits');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
figure;

























