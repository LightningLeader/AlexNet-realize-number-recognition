# Matlab搭建AlexNet实现手写数字识别

## 环境

- Matlab 2020a
- Windows10

## 内容

使用Matlab对MNIST数据集进行预处理，搭建卷积神经网络进行训练，实现识别手写数字的任务。在训练过程中，每隔30个batch输出一次模型在验证集上的准确率和损失值。在训练结束后会输出验证集中每个数字的真实值、网络预测值和判定概率，并给出总的识别准确率。

## 步骤

### 准备MNIST数据集

为了方便进行测试，本次只选用500张MNIST数据集，每个数字50张。

**[数据集下载地址](https://pan.baidu.com/s/1COuYnUrywAbCqWoDMkwQig?pwd=af6n)  提取码：af6n**

下载数据集后并解压，为每个数字创建单独文件夹并将该数字的所有图片放在对应的文件夹下，如图1所示。

![图1 将图片按数字分类](https://github.com/LightningLeader/LightningLeader.github.io/blob/master/posts/21/1.png)

手动分类结束后每个文件夹中应有50张图片。

### 数据预处理

```matlab
% 加载数据集
imds = imageDatastore(...
    "./data",...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames');
```

使用`imageDatastore`加载数据集。第一个参数填写数据集路径。由于本次实验data目录下含有子文件夹所以`IncludeSubfolders`需要指定为true。`LabelSource`表示标签来源，这里使用文件夹名字来代表标签。

```matlab
  ImageDatastore - 属性:

                       Files: {
                              'D:\data\0\0_1.bmp';
                              'D:\data\0\0_10.bmp';
                              'D:\data\0\0_11.bmp'
                               ... and 497 more
                              }
                     Folders: {
                              'D:\data'
                              }
                      Labels: [0; 0; 0 ... and 497 more categorical]
    AlternateFileSystemRoots: {}
                    ReadSize: 1
      SupportedOutputFormats: [1×5 string]
         DefaultOutputFormat: "png"
                     ReadFcn: @readDatastoreImage

```

上面内容为执行imageDatastore后返回变量的属性。可以看出已经成功将数据集读入并对每张图片进行label处理。

由于每个数字有50张图像，因此本次实验每个数字选用30张进行训练，另20张进行验证。使用splitEachLabel进行划分，得到训练集和验证集。

```matlab
% 数据打乱
imds = shuffle(imds);

% 划分训练集和验证集。每一个类别训练集有30个，验证集有20个
[imdsTrain,imdsValidation] = splitEachLabel(imds, 30);
```

使用shuffle进行数据打乱。得到的imdsTrain和imdsValidation分别有300和200张图片。

```matlab
% 将训练集与验证集中图像的大小调整成与输入层的大小相同
augimdsTrain = augmentedImageDatastore([28 28],imdsTrain);
augimdsValidation = augmentedImageDatastore([28 28],imdsValidation);
```

### 定义网络模型

```matlab
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
```

使用上面的代码即可构建AlexNet模型。

```matlab
% 对构建的网络进行可视化分析
lgraph = layerGraph(mynet);
analyzeNetwork(lgraph)
```

<img src="https://github.com/LightningLeader/LightningLeader.github.io/blob/master/posts/21/2.png" alt="2" style="zoom:50%;" />

### 定义训练超参数

```matlab
% 配置训练选项   
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...    
    'MaxEpochs',100, ...               
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress'); 
```

本次实验选用sgdm作为优化器，初始学习率设置为0.001，最大迭代次数为100，每次迭代都会打乱数据，每隔30个batch进行一次验证。

### 网络训练和预测

```matlab
% 对网络进行训练
net = trainNetwork(augimdsTrain, mynet, options); 

% 将训练好的网络用于对新的输入图像进行分类，得到预测结果和判定概率
[YPred, err] = classify(net, augimdsValidation);
```

其中，YPred是存放网络对验证集预测结果的数组，err存放着每个数字的判定概率。

![3](https://github.com/LightningLeader/LightningLeader.github.io/blob/master/posts/21/3.png)

```matlab
% 打印真实数字、预测数字、判定概率和准确率
YValidation = imdsValidation.Labels;
for i=1:200
fprintf("真实数字：%d  预测数字：%d", double(YValidation(i,1))-1, double(YPred(i, 1))-1);
fprintf("  判定概率：%f\n", max(err(i, :)));
end
```

运行上面代码即可打印相关结果。

```matlab
... ...
真实数字：4  预测数字：4  判定概率：0.814434
真实数字：0  预测数字：0  判定概率：0.657829
真实数字：8  预测数字：8  判定概率：0.874560
真实数字：0  预测数字：0  判定概率：0.988826
真实数字：6  预测数字：6  判定概率：0.970034
... ...
真实数字：5  预测数字：5  判定概率：0.806220
真实数字：4  预测数字：4  判定概率：0.938233
真实数字：7  预测数字：7  判定概率：0.906994
真实数字：7  预测数字：7  判定概率：0.837794
真实数字：6  预测数字：6  判定概率：0.951572
真实数字：6  预测数字：1  判定概率：0.415834
真实数字：5  预测数字：5  判定概率：0.789031
真实数字：2  预测数字：2  判定概率：0.363526
真实数字：7  预测数字：7  判定概率：0.930049

准确率：0.880000
```









