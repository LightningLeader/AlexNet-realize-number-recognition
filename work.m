clear;
clc;
% 加载数据集
imds = imageDatastore(...
    "./data",...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames');

% 数据打乱
imds = shuffle(imds);

% 划分训练集和验证集。每一个类别训练集有30个，验证集有20个
[imdsTrain,imdsValidation] = splitEachLabel(imds, 30);

% 将训练集与验证集中图像的大小调整成与输入层的大小相同
augimdsTrain = augmentedImageDatastore([28 28],imdsTrain);
augimdsValidation = augmentedImageDatastore([28 28],imdsValidation);

% 构建alexnet卷积网络 
alexnet = [
    imageInputLayer([56,56,1], 'Name', 'Input')
    convolution2dLayer([11,11],48,'Padding','same','Stride',4, 'Name', 'Conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'Relu_1')
    maxPooling2dLayer(3,'Padding','same','Stride',2, 'Name', 'MaxPooling_1')
    convolution2dLayer([5,5],128,'Padding',2,'Stride',1, 'Name', 'Conv_2')
    batchNormalizationLayer('Name', 'BN_2')
    reluLayer('Name', 'Relu_2')
    maxPooling2dLayer(3,'Padding','same','Stride',2, 'Name', 'MaxPooling_2')
    convolution2dLayer([3 3],192,'Padding',1,'Stride',1, 'Name', 'Conv_3')
    batchNormalizationLayer('Name', 'BN_3')
    reluLayer('Name', 'Relu_3')
    convolution2dLayer([3 3],192,'Padding',1,'Stride',1, 'Name', 'Conv_4')
    batchNormalizationLayer('Name', 'BN_4')
    reluLayer('Name', 'Relu_4')
    convolution2dLayer([3 3],128,'Stride',1,'Padding',1, 'Name', 'Conv_5')
    batchNormalizationLayer('Name', 'BN_5')
    reluLayer('Name', 'Relu_5')
    maxPooling2dLayer(3,'Padding','same','Stride',2, 'Name', 'MaxPooling_3')
    fullyConnectedLayer(4096, 'Name', 'FC_1')
    reluLayer('Name', 'Relu_6')
    fullyConnectedLayer(4096, 'Name', 'FC_2')
    reluLayer('Name', 'Relu_7')
    fullyConnectedLayer(10, 'Name', 'FC_3')    % 将新的全连接层的输出设置为训练数据中的种类
    softmaxLayer('Name', 'Softmax')            % 添加新的Softmax层
    classificationLayer('Name', 'Output') ];   % 添加新的分类层

mynet = [
    imageInputLayer([28, 28, 1], 'Name', 'Input')
    convolution2dLayer([5,5],50,'Name', 'Conv')
    tanhLayer('Name', 'tanh')
    averagePooling2dLayer(12,'Name', 'avgPooling')
    fullyConnectedLayer(100, 'Name', 'FC_1')
    fullyConnectedLayer(10, 'Name', 'FC_2')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
];
% 对构建的网络进行可视化分析
lgraph = layerGraph(alexnet);
analyzeNetwork(lgraph)

% 配置训练选项   
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...    
    'MaxEpochs',100, ...               
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');                

% 对网络进行训练
net = trainNetwork(augimdsTrain, mynet, options); 

% 将训练好的网络用于对新的输入图像进行分类，得到预测结果和判定概率
[YPred, err] = classify(net, augimdsValidation);

% 打印真实数字、预测数字、判定概率和准确率
YValidation = imdsValidation.Labels;
for i=1:200
    fprintf("真实数字：%d  预测数字：%d", double(YValidation(i,1))-1, double(YPred(i, 1))-1);
    fprintf("  判定概率：%f\n", max(err(i, :)));
end


accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf("\n准确率：%f\n", accuracy);
