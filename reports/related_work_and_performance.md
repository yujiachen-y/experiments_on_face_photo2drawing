<center><h1>人脸照片画像转换评价方法及相关工作</h1></center>

<center><i>written by  <a href="http://yujiachen.top/">Yu, Jiachen</a></i></center>

本文主要关注用何种方法评价人脸照片画像转换的结果，并阐述对应评价方法的特性。之后在这些方法下，如何对数据集进行划分，如何设计实验步骤。并在上述基础上，测试相应的一些模型，结合测评结果分析模型的特点。

# 评价方法

## GAN评价方法

### fid
[1]中对各种GAN评价方法进行了详尽分析，认为fid[2]是目前所有评价方法中，比较合理的一个选择。fid的结果和人类对图片的观感有一定的相关性，可以检测出模式崩塌，也对损坏图像敏感[2]，而且其计算效率也较高。
fid的计算公式如下：
$$
FID(r, g) = \left \| \mu_r - \mu_g \right \|_2^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{\frac{1}{2}})
$$
其中$(\mu_r, \Sigma_r)$和$(\mu_g, \Sigma_g)$分别是真实数据分布和生成数据分布的均值和协方差矩阵。

但是fid也有着一些缺陷，**fid假设高维空间中的特征是高斯分布的，这个条件有时不会得到满足[1]。**fid需要Inception-v3[3]作为特征抽取器，因为Inception是训练在ImageNet[4]上的神经网络，其目标是分类，所以我们无法确定fid对某些不自然的图像变形会有何反应，也无法说明fid用来评价一些与ImageNet领域无关任务（比如人脸生成和数字、字体生成）的合理性[1]。

### 分类器双样本测试（Classifier Two-sample Tests(C2ST)）
C2ST认为，如果两个分布是相等的，那么从这两个分布中的任一个分布中拿出任一个图片，并且根据两个分布设置一个对应的2类别分类器，这个图片被这个分类器分到两个类别的概率是相等的。如果概率不相等，那么这两个分布就不是相等的分布。
实际操作中，可以把正负样本相等的测试集随机分成测试训练集和测试测试集，用测试训练集训练分类器，测试测试集计算分类准确率。根据测试准确率计算p值，来决定是否接受两个分布是相等的这个零假设。
具体到GAN的评测上，我们还可以根据分类的结果判断出模型是过拟合还是欠拟合。此时我们从**训练集**中拿出n张真实图像[6]，然后让模型生成n张生成图像，如果生成图像被分类到生成图像上的概率为0%，那么说明模型记住了真实图像，导致生成图像和真实图像完全相似，此时发生了过拟合。如果生成图像被分类到生成图像上的概率为100%，说明模型生成的图像和真实图像完全不相似，此时发生了欠拟合。
[5]认为可以用1近邻分类器（1NN-classifier）作为分类器，其有着可以判断模型是否发生过拟合，数值上易于比较，计算效率高等优势。
[5]用如下方法计算（计算过程由自己根据[6]整理）：
1. 先用当前的G网络生成图片，然后从训练集中随机采样出一些图片。
2. 随后对每张图片存下对应的像素空间（输入），卷积空间（卷积层输出），类别分数（fc层输出），类别概率（softmax输出）
3. 对大小为n的mini_batch，计算出真实数据和生成数据之间和各自的n\*n距离矩阵，n\*n矩阵的计算方法为(x\*\*2 - 2x\*y + y\*\*2)
4. 根据距离进行1nn分类，即不算自己和自己的距离，看和自己距离最近的特征的标签是哪一个，以此计算一系列tp、fp等分类数值

## 人脸质量评价方法

# 参考
[1] Borji A. Pros and Cons of GAN Evaluation Measures[J]. arXiv preprint arXiv:1802.03446, 2018.
[2] Heusel M, Ramsauer H, Unterthiner T, et al. GANs trained by a two time-scale update rule converge to a Nash equilibrium[J]. arXiv preprint arXiv:1706.08500, 2017.
[3] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.
[4] Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical image database[C]//Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. Ieee, 2009: 248-255.
[5] Xu Q, Huang G, Yuan Y, et al. An empirical study on evaluation metrics of generative adversarial networks[J]. arXiv preprint arXiv:1806.07755, 2018.
[6] xuqiantong/GAN-Metrics[EB/OL]. GitHub, 2018. (2018)[2018 -10 -27]. https://github.com/xuqiantong/GAN-Metrics.