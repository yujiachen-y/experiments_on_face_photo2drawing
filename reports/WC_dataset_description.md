*written by  [Jiachen YU](http://yujiachen.top/)*

# README信息
## 统计
有252个对象，以及来自这252个对象的6042张漫画和5974张图片。
每张图片有不同的分辨率，以及有的图像是灰度图像，有的图象是RGB图像。
每张图像上都有17个脸部landmarks
数据集网站：https://cs.nju.edu.cn/rl/WebCaricature.htm

## 文件结构
- OriginalImages：目录下是各个人名的子目录，每个人名子目录下C开头的照片是卡通画像，P开头的是真实照片
- FacialPoints：以txt的形式存着每张照片的landmark，每个landmark点两个数字，前面代表着x轴坐标，后面代表着y轴坐标 **（如何标记这些点？实际上访问矩阵的时候是不是先访问y再访问x？）**，landmark的格式如下
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

## 参考
[1] Jing Huo, Wenbin Li, Yinghuan Shi, Yang Gao and Hujun Yin, "WebCaricature: A Benchmark for Caricature Recognition", British Machine Vision Conference, 2018.
[2] Jing Huo, Yang Gao, Yinghuan Shi, Hujun Yin, "Variation Robust Cross-Modal Metric Learning for Caricature Recognition", ACM Multimedia Thematic Workshops, 2017: 340-348.

# WebCaricature: a benchmark for caricature recognition
## 对论文的小结

论文提到了现有的几个数据集：
![](reports\elements\existing_datasets_relating_to_caricature.png)

- 照片的landmark是用face++的api标的
- 画像的landmark是手工标的，没说怎么具体标记的