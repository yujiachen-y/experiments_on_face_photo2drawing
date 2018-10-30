<center><h1>人脸照片画像转换评价方法及相关工作</h1></center>

<center><i>written by  <a href="http://yujiachen.top/">Yu, Jiachen</a></i></center>

本文主要关注用何种方法评价人脸照片画像转换的结果，并阐述对应评价方法的特性。之后在这些方法下，如何对数据集进行划分，如何设计实验步骤。并在上述基础上，测试相应的一些模型，结合测评结果分析模型的特点。

# 评价方法

## GAN评价方法
### fid
[1]中对各种GAN评价方法进行了详尽分析，认为fid[2]是目前所有评价方法中，比较合理的一个选择。fid的结果和人类对图片的观感有一定的相关性，可以检测出训练工程中的异常，也对损坏图像敏感[2]，而且其计算效率也较高。

fid的计算公式如下：
$$
FID(r, g) = \left \| \mu_r - \mu_g \right \|_2^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{\frac{1}{2}})
$$
其中$(\mu_r, \Sigma_r)$和$(\mu_g, \Sigma_g)$分别是真实数据分布和生成数据分布的均值和协方差矩阵。

但是fid也有着一些缺陷，**fid假设高维空间中的特征是连续多元高斯分布的，这个条件有时不会得到满足[1]。** fid需要Inception-v3[3]作为特征抽取器（当然需要的话，也可以把Inception-v3替换为其他网络），因为Inception是训练在ImageNet[4]上的神经网络，其目标是分类，所以我们无法确定fid对某些不自然的图像变形会有何反应，也无法说明fid用来评价一些与ImageNet[4]领域无关任务（比如人脸生成和数字、字体生成）的合理性[1]。

[1]中认为因为Inception-v3[3]是由ImageNet[4]训练得来，所以用于评估其他任务的合理性难以说明，个人觉得ImageNet[4]的数据量和类别数目是非常大的，以此训练得来的网络对图像的特征抽取能力肯定不会很弱。

### 分类器双样本测试（Classifier Two-sample Tests(C2ST)）
C2ST认为，如果两个分布是相等的，那么从这两个分布中的任一个分布中拿出任一个图片，并且根据两个分布设置一个对应的2类别分类器，这个图片被这个分类器分到两个类别的概率是相等的。如果概率不相等，那么这两个分布就不是相等的分布。

实际操作中，可以把正负样本相等的测试集随机分成测试训练集和测试测试集，用测试训练集训练分类器，测试测试集计算分类准确率。根据测试准确率计算p值，来决定是否接受两个分布是相等的这个零假设。

具体到GAN的评测上，我们还可以根据分类的结果判断出模型是过拟合还是欠拟合。此时我们从**训练集** 中拿出n张真实图像[6]，然后让模型生成n张生成图像，如果生成图像被分类到生成图像上的概率为0%，那么说明模型记住了真实图像，导致生成图像和真实图像完全相似，此时发生了过拟合。如果生成图像被分类到生成图像上的概率为100%，说明模型生成的图像和真实图像完全不相似，此时发生了欠拟合。

[5]认为可以用1近邻分类器（1NN-classifier）作为C2ST的分类器，根据[5]的实验结果，1近邻分类器在实验中表现出了可以检测出训练时发生的多种异常情况，计算效率高，不过过多受到数据集大小干扰等特点。另外，1近邻分类器的评价分数只在[0, 1]之间，数值上的比较和解释性更强。

[5]用如下方法计算（计算过程由自己根据[6]整理）：

1. 先用当前的G网络生成图片，然后从训练集中随机采样出一些图片。
2. 随后对每张图片存下对应的像素空间（输入），卷积空间（卷积层输出），类别分数（fc层输出），类别概率（softmax输出）
3. 对大小为n的mini_batch，计算出真实数据和生成数据之间和各自的n\*n距离矩阵，n\*n矩阵的计算方法为$(x^2 - 2x \ast y + y^2)$
4. 根据距离进行1nn分类，即不算自己和自己的距离，看和自己距离最近的特征的标签是哪一个，以此计算一系列tp、fp等分类数值
    实验结果表明用34层的ResNet作为特征抽取器，在卷积空间（卷积层输出结果）中，1近邻分类器的表现最好。另外，把34层的ResNet替换为VGG或者Inception模型也可以得到相似的实验结果[5]。

### Wasserstein Distance
推土机距离(Earth Mover distance)，表示在高维空间中，把一个分布拟合进另一个分布所需要的代价。计算公式为
$$
W(\Bbb P_r, \Bbb P_g)= \operatorname{inf}\limits_{\gamma\sim\prod(\Bbb P_r, \Bbb P_g)}\Bbb E_{(x, y)\sim\gamma}[\left \| x - y \right \|]
$$
因为Wasserstein Distance的计算复杂度过高，外加如何在计算前对高维特征进行预处理也需要考虑，以及[1]和[5]都没有推荐使用该方法，所以目前不使用这个方法作为评价方法。

## 人脸识别评价方法
人脸识别的评价方法可以使用预训练好的人脸识别网络来判断生成的人脸识别度如何。

人脸识别度的评价方法已经非常成熟，因为此任务对生成的人脸没有非常严格的要求，这里可以直接用[8]中的评价方法：对verification（1:1，判断两张照片是不是同一个人）用VR@FAR=0.1%，VR@FAR=1%和AUC来评定；对identification（1:n，判断1张照片在不在一个底库中，是底库中的谁）用Rank-1和Rank-10来判定。

同时，[8]中也提供了使用这些评价方法在一些模型上得到的指标，其中使用的神经网络模型是VGG-face[9]。

## 总结
因为人脸照片-画像转化工作需要生成真实图像的同时保存身份信息，所以需要GAN评价方法和人脸识别评价方法两个一起评测。对GAN可以使用fid方法评测，考虑到目前数据集过小，可以同时用1NN分类器进行评测，另外对生成的人脸，可以用人脸识别评价方法来判断生成人脸的身份特征被保留的程度。

# 相关工作
本段枚举一些相关工作被应用在人脸照片-画像转换时的表现。因为时间问题，下次提交报告时补充。

# 参考
[1] Borji A. Pros and Cons of GAN Evaluation Measures[J]. arXiv preprint arXiv:1802.03446, 2018.

[2] Heusel M, Ramsauer H, Unterthiner T, et al. GANs trained by a two time-scale update rule converge to a Nash equilibrium[J]. arXiv preprint arXiv:1706.08500, 2017.

[3] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.

[4] Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical image database[C]//Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. Ieee, 2009: 248-255.

[5] Xu Q, Huang G, Yuan Y, et al. An empirical study on evaluation metrics of generative adversarial networks[J]. arXiv preprint arXiv:1806.07755, 2018.

[6] xuqiantong/GAN-Metrics[EB/OL]. GitHub, 2018. (2018)[2018 -10 -27]. https://github.com/xuqiantong/GAN-Metrics.

[7] Salimans T, Goodfellow I, Zaremba W, et al. Improved techniques for training gans[C]//Advances in Neural Information Processing Systems. 2016: 2234-2242.

[8] Jing Huo, Wenbin Li, Yinghuan Shi, Yang Gao and Hujun Yin, "WebCaricature: A Benchmark for Caricature Recognition", British Machine Vision Conference, 2018.

[9] Parkhi O M, Vedaldi A, Zisserman A. Deep face recognition[C]//BMVC. 2015, 1(3): 6.

# 参考文献说明
## 参考1：Pros and Cons of GAN Evaluation Measures
该文分析了24种定量方法和5种定性方法，并列出了各种方法在评价GAN模型时的优缺点、注意事项和相关参考文献。以此希望为GAN的评价工作提供一个详尽的参考。

## 参考5：An empirical study on evaluation metrics of generative adversarial networks
该文以实验结果为结论，设计了Mode Collapsing、Mode Dropping、图像变形、数据集过小、数据集过大、模型过拟合等情况发生时，The Inception Score, Kernel MMD, Wasserstein distance, Frechet Inception Distance, 1-NN classifier这6个评价方法在实验中的表现。

自己看了这篇文章后，才发现之前一直对Mode Collapsing、Mode Dropping、模型过拟合这3个现象有不理解和混淆之处，这里可以记录一下3个现象的区别

- Mode Collapsing：模式坍塌，指真实分布中一些相似的模式被generator理解成了一个模式，generator会把这些模式替代为一个"平均"的模式，[5]在实验中把数据集中的图片进行聚类，把一个聚类中的所有图片用聚类中心代替，以此来模拟Mode Collapsing。
- Mode Dropping：模式丢失，指真实分布中一些可能过于复杂的模式没有被generator学习，generator不会生成符合某一模式的图片，[5]在实验中把数据集中的图片进行聚类，选择几个聚类中，删去这些聚类中的所有图片，以此来模拟Mode Dropping。
- **个人理解，上述的Mode Collapsing 和Mode Dropping 都是模型学习欠拟合的表现。** 
- Overfitting: 模型过拟合，指的是模型的学习能力太强，而训练时使用的数据集又太小，使得模型记住了训练集中的数据，此时模型在训练集上的表现会很好，但是在测试集上不会有很好的表现。