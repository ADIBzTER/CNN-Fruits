datasetPath = fullfile('Fruits');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Show random images
figure;
perm = randperm(200, 20);
for i = 1:20
    subplot(4, 5, i);
    imshow(imds.Files{perm(i)});
end

%% Count labels
labelCount = countEachLabel(imds)

%% Show image size
img = readimage(imds, 1);
size(img)

%% Split datastore
numTrainFiles = 75;
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

%% Define Network Architecture for CNN
layers = [
    % [height width channel]
    imageInputLayer([100 100 3])
    
    % (filterSize, numOfFilters, name, value)
    % numOfFilters => number of neuron
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    % (poolSize, name, value)
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    % (outputSize)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
];

%% Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train Network
net = trainNetwork(imdsTrain, layers, options);

%% Input size of input layer
inputSize = net.Layers(1).InputSize

%% Initialize testing image size
I = imread('testOrange.jpg');

%% Show testing image
figure
imshow(I)

%% Resize testing image
I = imresize(I, inputSize(1:2));

%% Show image size
size(I)

%% Classifying test image
[label, scores] = classify(net, I);
label

%% Show Predicted Probability
figure
imshow(I)
classNames = net.Layers(end).ClassNames;
title(string(label) + ", " + num2str(100 * scores(classNames == label), 3) + "%");

%% Build Graph

[~, idx] = sort(scores, 'descend');
idx = idx(4:-1:1);
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Predictions')
xlabel('Probability')
yticklabels(classNamesTop)



