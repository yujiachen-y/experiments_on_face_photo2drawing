<center><h1>毕设数据集处理报告</h1></center>

<center><i>written by  <a href="http://yujiachen.top/">Yu, Jiachen</a></i></center>

# README信息
## 统计
有252个对象，以及来自这252个对象的6042张漫画和5974张图片。
每张图片有不同的分辨率，以及有的图像是灰度图像，有的图象是RGB图像。
每张图像上都有17个脸部landmarks
数据集网站：https://cs.nju.edu.cn/rl/WebCaricature.htm

## 文件结构
- OriginalImages：目录下是各个人名的子目录，每个人名子目录下C开头的照片是卡通画像，P开头的是真实照片
- FacialPoints：以txt的形式存着每张照片的landmark，每个landmark点两个数字，前面代表着x轴坐标，后面代表着y轴坐标，不过注意一张图片被读进的坐标如下，landmark的格式如下
  ![](elements/cv_image.png)
  ![landmarks_description](https://cs.nju.edu.cn/rl/Caricature_files/example/Landmarks_black.jpg)

| Landmark | Meaning | Landmark | Meaning |
| ------ | ------ | ------ | ------|
| 1 | Contour(Top of Hairline) | 10 | Right corner of left eye |
| 2 | Contour(Center of left ear) | 11 | Left corner of right eye |
| 3 | Contour(Chin) | 12 | Right corner of right eye |
| 4 | Contour(Center of right ear) | 13 | Nose tip |
| 5 | Left corner of left eyebrow | 14 | Mouth upper lip top |
| 6 | Right corner of left eyebrow | 15 | Mouth left corner |
| 7 | Left corner of right eyebrow | 16 | Mouth lower lip bottom |
| 8 | Right corner of right eyebrow | 17 | Mouth right corner |
| 9 | Left corner of left eye |  |  |
- Filenames: 以txt的形式存着每个人对应的漫画图片和真实照片的文件名。
- EvaluationProtocols：存放着各个评价方案在各种情况下所需用到的各种数据。

## 命名
略过不谈。

## 评价方案
对所有4个评价方案的实验设置，有2种设置(two views)。设置1(View 1)中的数据只应该用来调整参数，设置2(View 2)中的数据用来评估模型的性能。

**数据集中的说明在此不翻译**

### 严格漫画验证
文件夹"EvaluationProtocols\FaceVerification\Restricted"下有3个文件。
"RestrictedView1_DevTrain.txt" 和 "RestrictedView1_DevTest.txt"属于设置1，只能用来调整参数。
"RestrictedView2.txt"可以用来做模型评估。

In both "RestrictedView1_DevTrain.txt" and "RestrictedView1_DevTest.txt", the first line gives the number of both matched and mismatched pairs. For example, 100 means there are 100 matched pairs and 100 mismatched pairs. The following lines gives the matched and mismatched pairs. Each line contains a matched pair or a mismatched pair.

For example, "Adele Laurie Blue Adkins	C00001	P00001" means the caricature filename of "C00001" of "Adele Laurie Blue Adkins" and the photo filename of "P00001" of "Adele Laurie Blue Adkins" are from a matched pair.
"Elizabeth Taylor	C00039	Jennifer Aniston	P00006" means the caricature filename of "C00039" of "Elizabeth Taylor" and the photo filename of "P00006" of "Jennifer Aniston" are from a mismatched pair.

在用设置1的实验确定了参数以后，我们用设置2进行性能评估。
In "RestrictedView2.txt", there are ten folds. For each fold, the first line gives the number of matched and mismatched pairs, followed by information of matched and mismatched pairs. For result reporting, 9 folds of the pairs should be used for training and the remaining fold for testing. This process should be repeated for ten times.

### 非严格漫画验证
严格漫画验证和非严格漫画验证的主要区别是，在严格验证下，只有数据集指定的图片对可以用来训练。但是，在非严格验证下，实验人员可以自主构建图片对来进行训练。

文件夹"EvaluationProtocols\FaceVerification\UnRestricted"下有3个文件。
"UnRestrictedView1_DevTrain.txt" 和 "UnRestrictedView1_DevTest.txt"属于设置1，只能用来调整参数。
"UnRestrictedView2.txt"可以用来做模型评估。

In both "UnRestrictedView1_DevTrain.txt" and "UnRestrictedView1_DevTest.txt", the first line gives the number of subjects that can be used for training or testing. For example, 20 means there are 20 subjects that can be used for training or testing. The following lines give the names of the subjects, their caricature numbers and photo numbers. Each line contains a name and two numbers, the first number is the subject's caricature number, and the second number is the subject's photo number.

For example, "Jeff Bridges	35	25" means Jeff Bridges has 35 caricatures and 25 caricatures. All these images can be used for either training or testing.

In "UnRestrictedView2.txt", there are ten folds. For each fold, the first line also gives the number of subjects in that fold, followed by a few lines, with each line containing a name of the subject, its number of caricatures and its number of photos. For result reporting, 9 folds of the subjects should be used for training and the remaining fold for testing. On the testing fold, all the same-person and different-person pairs should be constructed for result reporting. This process should be repeated for ten times. 

### 漫画-照片识别
Files of View 1 include:
-"EvaluationProtocols\FaceIdentification\FR_Train_dev.txt" 
-"EvaluationProtocols\FaceIdentification\C2P\FR_Gallery_C2P_dev.txt" 
-"EvaluationProtocols\FaceIdentification\C2P\FR_Probe_C2P_dev.txt"
View 1 is used for parameter tuning. "FR_Train_dev.txt" can be used for training. "FR_Gallery_C2P_dev.txt" and "FR_Probe_C2P_dev.txt" can be used for evaluation while tuning parameters. 
"FR_Train_dev.txt" contains a few lines, with each line containing a name of the subject, its number of caricatures and its number of photos. All the caricatures and photos of the subject can be used for training. For evaluation of models during parameter tuning stage, "FR_Gallery_C2P_dev.txt" contains filenames of images that can be used as gallery. It is a set of photos. "FR_Probe_C2P_dev.txt" contains filenames of images that can be used as probes. It is a set of caricatures.  

Files of View 2 include:
-"EvaluationProtocols\FaceIdentification\FR_TrainX.txt"
-"EvaluationProtocols\FaceIdentification\C2P\FR_Gallery_C2PX.txt"
-"EvaluationProtocols\FaceIdentification\C2P\FR_Probe_C2PX.txt" 
where 'X' ranges from 1 to 10, corresponding to 10 folds. A "FR_TrainX.txt" contains a few lines, with each line containing a name of the subject, its number of caricatures and its number of photos. All the caricatures and photos of the subject can be used for training. Then the corresponding "FR_Gallery_C2PX.txt", "FR_Probe_C2PX.txt" should be used for testing. This process should be done ten times for result reporting.

### 照片-漫画识别
Files of View 1 include:
-"EvaluationProtocols\FaceIdentification\FR_Train_dev.txt" 
-"EvaluationProtocols\FaceIdentification\P2C\FR_Gallery_P2C_dev.txt" 
-"EvaluationProtocols\FaceIdentification\P2C\FR_Probe_P2C_dev.txt"
View 1 is used for parameter tuning. "FR_Train_dev.txt" can be used for training. "FR_Gallery_P2C_dev.txt" and "FR_Probe_P2C_dev.txt" can be used for evaluation while tuning parameters. 
"FR_Train_dev.txt" contains a few lines, with each line containing a name of the subject, its number of caricatures and its number of photos. All the caricatures and photos of the subject can be used for training. For evaluation of models during parameter tuning stage, "FR_Gallery_P2C_dev.txt" contains filenames of images that can be used as gallery. It is a set of caricatures. "FR_Probe_P2C_dev.txt" contains filenames of images that can be used as probes. It is a set of photos.  

Files of View 2 include:
-"EvaluationProtocols\FaceIdentification\FR_TrainX.txt"
-"EvaluationProtocols\FaceIdentification\P2C\FR_Gallery_P2CX.txt"
-"EvaluationProtocols\FaceIdentification\P2C\FR_Probe_P2CX.txt"
where 'X' ranges from 1 to 10, corresponding to 10 folds. A "FR_TrainX.txt" contains a few lines, with each line containing a name of the subject, its number of caricatures and its number of photos. All the caricatures and photos of the subject can be used for training. Then the corresponding "FR_Gallery_P2CX.txt", "FR_Probe_P2CX.txt" should be used for testing. This process should be done ten times for result reporting.

## 注意：
1. 该数据集只能用于学术目的
2. 请不要重新布置或修改数据集
3. 如您有漏洞报告，或关于使用数据集的疑问或建议，请联系Jing Huo (huojing@nju.edu.cn)

# WebCaricature: a benchmark for caricature recognition
接下来写一下对数据集论文的小结，参考的论文是[1]的v4版本

### 介绍
论文提到了现有的几个数据集：
![](elements\\existing_datasets_relating_to_caricature.png)

### 数据采集
- 数据的收集是通过google image搜索得来的
- 照片的landmark是用face++的api标的
- **画像的landmark是手工标的，但文章没说标记的具体流程和消耗的资源**

### 评价方案
- 文章说明了使用数据集时，哪些数据可以用在哪些步骤中
- 文章建议Caricature Verification任务的评测指标是ROC的AUC、FAR=0.1%时的验证正确率和FAR=1%时的验证正确率
- 文章建议Caricature Identification任务的评测指标是CMC中的rank-1和rank-10
- 文章建议的评测方法是用10-fold的交叉验证，对10次得到的最终结果取平均得到最终评测结果
- **文章没有给出制定上述评价方案的依据**，自己读完了论文之后，感觉数据集对我仍像黑盒一样未知

### 识别框架
#### landmark标记
画像是用手工标记的，落实到框架里如何标记目前没有解决方案，**自己感觉画像的landmark标记可以单独作为一个课题拿出来做**

#### 脸部对齐
文章采用了三个方案，具体的操作和参数见论文
1 根据两个眼睛的关键点位置来完成alignment（根据[3]里面的实验结果，自己打算用5个或者3个landmark做相似变换来align）
2 根据17个landmark的位置得到17个patch，从17个patch中抽取特征
3 根据脸外围的4个轮廓点，把脸裁出来

#### 匹配算法
作者多次提到了自己的跨模态尺度学习方法[2]，有空看一看。不过我们的工作是图像翻译相关的，所以可能帮助不会很大。

## Balseline表现
具体数据见原文。
文中说到了第2中脸部对齐方法在传统模型里的表现最好，但是不适合应用在深度学习里，所以就没做相关实验，但是文中后面明确说了实验没有fine tune vgg face，而是直接用的vgg face的原参数，我觉得是论文作者不想花时间重新训一个模型所以没有验证第2个方法的表现。

# 数据集处理
因为图像生成任务本身有一定难度，所以预处理数据集，来简化模型的任务也是一个需要仔细考虑的问题，目前打算有如下对数据集进行处理的打算。
- 人脸转正：根据landmark从原图中抠出人脸块，转正，然后考虑如何处理筛选出的数据
- 数据筛选：踢掉一些难度过高的数据，然后考虑如何处理筛选出的数据
- 不对数据集进行处理

## 人脸转正

首先选择了[7]的代码示例[8]中用到的5个点作为landmark，分别为

| Landmark Coordinate |      Meaning       |
| :-----------------: | :----------------: |
|  30.2946, 51.6963   |      Left eye      |
|  65.5318, 51.5014   |     Right eye      |
|  48.0252, 71.7366   |      Nose tip      |
|  33.5493, 92.3655   | Mouth left corner  |
|  62.7299, 92.2041   | Mouth right corner |

随后根据原有数据集中每个图像（画像和照片）的landmark，进行仿射变换(affine transformation)，并注意原有数据集中的landmark是没有左右眼的landmark的，这里我们在操作中取每个眼睛眼角landmark的平均值作为每个眼睛的landmark。

完成转正后的数据集大体如下：

![](elements\frontalization_dataset_example_v1.png)

## 数据筛选
模型初步并不打算处理较难的数据，所以需要一些方法来剔除过难数据。

### ~~剔除大角度数据~~ 

起初打算通过筛选大角度数据的方法来删除高难度数据，但是效果不好，加上后面找到了更合理的方法，放弃。

先根据参考[4]中的方法和参考[5]和[6]中的3-D人脸模型估算出每张图片中人脸的欧拉角度(pitch, yaw, roll)，欧拉角度的示意图如下：

![](elements\euler_angles.jpg)

评估图像中人脸欧拉角度的landmark如下：

|         Landmark         |         Landmark          |
| :----------------------: | :-----------------------: |
| Left corner of left eye  | Right corner of left eye  |
| Left corner of right eye | Right corner of right eye |
|         Nose tip         |       Contour(Chin)       |
|   Mouth upper lip top    |     Mouth left corner     |
|  Mouth lower lip bottom  |    Mouth right corner     |

也许在欧拉角度的计算方法不够优秀，目前算出人脸的欧拉角度还不够准确，外加上很多漫画图的人脸变形严重，偶尔会出现计算错误。

计算出欧拉角度后，数据集的划分策略如下：

1. yaw角度大于30度就直接剔除，效果还算不错，但也有正脸漫画被误判为侧脸的情况。

### 剔除landmark不规范数据

因为在前面引入了人脸转正，外加接下来的模型打算把人脸的landmark作为辅助信息来生成图片，因此用图形的landmark之间的位置关系来剔除数据是一个挺合理的方法。

首先我们检查图片转正后landmark之间的位置关系是否是一个正脸，具体的判断逻辑见代码，这个条件可以剔除大角度的侧脸。

然后检查转正后的landmark是否在被裁减的图片中，这个条件可以剔除掉过于扭曲的画像。

最后检查转中后的landmark和标准landmark[8]之间的距离是否在一个标准内（目前采用的方法是以两个landmark为半径的圆面积是否超过图片面积的5%），这个条件可以进一步降低网络学习的难度。

最后符合规范的数据有8550张图片。

下面是一些被剔除的landmark不规范数据：

![](elements\uncorrected_landmark_dataset_example_v1.png)

### ~~剔除和照片相似的画像~~

剔除和照片相似的画像可以进一步让网络理解我们需要其完成的任务，但是目前来看，只有通过人眼审核才能剔除此类数据，工作量过大，还需要标注工具的开发，因此目前先不进行这项工作，希望数据不会过于不纯。

## 数据集划分

预计明天处理。

# DatasetViewer

自己写了一个Django app来显示数据集，过段时间后会配置在服务器上。

![](elements\DatasetViewer_example_v1.png)

# 参考
[1] Jing Huo, Wenbin Li, Yinghuan Shi, Yang Gao and Hujun Yin, "WebCaricature: A Benchmark for Caricature Recognition", British Machine Vision Conference, 2018.

[2] Jing Huo, Yang Gao, Yinghuan Shi, Hujun Yin, "Variation Robust Cross-Modal Metric Learning for Caricature Recognition", ACM Multimedia Thematic Workshops, 2017: 340-348.

[3] Demystifying Face Recognition IV: Face-Alignment[EB/OL]. BLCV - Bartosz Ludwiczuk Computer Vision, 2018. (2018)[2018 -10 -01]. http://blcv.pl/static/2017/12/28/demystifying-face-recognition-iii-face-preprocessing/index.html.

[4] 基于Dlib和OpenCV的人脸姿态估计(HeadPoseEstimation) - 二极管具有单向导电性 - CSDN博客[EB/OL]. Blog.csdn.net, 2018. (2018)[2018 -10 -10]. https://blog.csdn.net/u013512448/article/details/77804161.

[5] T. Bolkart, S. Wuhrer, 3D Faces in Motion: Fully Automatic Registration and Statistical Analysis, Computer Vision and Image Understanding, 131:100–115, 2015

[6] T. Bolkart, S. Wuhrer, Statistical Analysis of 3D Faces in Motion, 3D Vision, 2013, pages 103-110

[7]Liu W, Wen Y, Yu Z, et al. Sphereface: Deep hypersphere embedding for face recognition[C]//The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017, 1: 1.

[8]Liu W, Wen Y, Yu Z, et al. wy1iu/sphereface[EB/OL]. GitHub, 2018. (2018)[2018 -10 -15]. https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m#L22.