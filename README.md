# 使用paddlex的分类功能 快速完成分类任务

## 一、项目背景介绍

### 分类任务是深度学习中一个很重要的内容，飞桨图像分类套件PaddleClas是飞桨为工业界和学术界所准备的一个图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。也是最成熟的一个领域。本文主要是介绍通过paddlex快速掌握分类任务。

## 二、数据介绍
275 种鸟类的数据集。39364张训练图像，1375张测试图像（每个物种5张）和1375张验证图像（每个物种5张。
所有图像均为jpg格式的224 X 224 X 3彩色图像。数据集包括训练集、测试集和验证集。每组包含 275 个子目录，每个鸟种一个。如果你使用 Keras ImageDataGenerator.flow from目录来创建你的训练、测试和有效数据生成器，数据结构会很方便。数据集还包括一个文件 Birds.csv。这个 cvs file 包含三列。filepaths 列包含图像文件的文件路径。labels 列包含与图像文件关联的类名。如果使用 df=pandas.birds 读入 data.csv 文件csv(data.csv) 将创建一个 Pandas 数据帧，然后可以将其拆分为训练df、测试df 和验证的df 数据，以创建您自己的数据划分为训练、测试和验证数据集。
注意：数据集中的测试和验证图像是手工选择的“最佳”图像，因此您的模型可能会使用这些数据集获得最高的准确度分数，而不是创建您自己的测试和验证集。然而，就看不见的图像的模型性能而言，后一种情况更准确。
 
网址链接：https://www.kaggle.com/gpiosenka/100-bird-species


## 三、模型介绍

### 1.定义训练/验证图像处理流程transforms
```
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])
```

### 2.定义dataset加载图像分类数据集

```
import paddlex as pdx
train_dataset = pdx.datasets.ImageNet(
    data_dir='/home/aistudio/Bird_Dataset/birds/',
    file_list='/home/aistudio/Bird_Dataset/birds/train_list.txt',
    label_list='/home/aistudio/Bird_Dataset/birds/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='/home/aistudio/Bird_Dataset/birds/',
    file_list='/home/aistudio/Bird_Dataset/birds/val_list.txt',
    label_list='/home/aistudio/Bird_Dataset/birds/labels.txt',
    transforms=eval_transforms)
```


## 四、模型训练
### 1.使用MobileNetV3_small_ssld模型开始训练
```
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_small(num_classes=num_classes)
model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small',
            use_vdl=True)
```
### 2.训练过程使用VisualDL查看训练指标变化
训练过程中，模型在训练集和验证集上的指标均会以标准输出流形式输出到命令终端。当用户设定use_vdl=True时，也会使用VisualDL格式将指标打点到save_dir目录下的vdl_log文件夹，在终端运行如下命令启动visualdl并查看可视化的指标变化情况。
```
visualdl --logdir output/mobilenetv3_small_ssld --port 8001
```
服务启动后，通过浏览器打开https://0.0.0.0:8001或https://localhost:8001即可。


## 五、模型评估
### 1.训练速度
![image.png](attachment:301cd525-4f68-4485-b5bc-fdcebe12b782.png)
![image.png](attachment:2bae04d4-5c4f-42d0-9aa4-5e706a5a06ff.png)
可以看到275个分类，几分钟即可完成
### 2.检测结果
```
import paddlex as pdx
model = pdx.load_model('output/mobilenetv3_small/best_model')
result = model.predict('/home/aistudio/Bird_Dataset/birds/test/AFRICAN_CROWNED_CRANE/1.jpg')
print("Predict Result: ", result)
```
结果如下：
![image.png](attachment:fc8b4c2e-8190-4cff-a793-f4367077c7a6.png)
精确度高


