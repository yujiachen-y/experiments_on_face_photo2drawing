<center><h1>人脸照片画像转换评价方法及相关工作</h1></center>

<center><i>written by  <a href="http://yujiachen.top/">Yu, Jiachen</a></i></center>

本文主要关注用何种方法评价人脸照片画像转换的结果，并阐述对应评价方法的特性。之后在这些方法下，如何对数据集进行划分，如何设计实验步骤。并在上述基础上，测试相应的一些模型，结合测评结果分析模型的特点。

# 评价方法

## GAN评价方法

### fid
[1]中对各种GAN评价方法进行了详尽分析，认为fid[2]是目前所有评价方法中，比较合理的一个选择。fid的结果和人类对图片的观感有一定的相关性，可以检测出模式崩塌，也对损坏图像敏感[2]，而且其计算效率也较高。
fid的计算公式如下：
！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
但是fid也有着一些缺陷，**fid假设高维空间中的特征是高斯分布的，这个条件有时不会得到满足[1]。**fid需要Inception-v3[3]作为特征抽取器，因为Inception是训练在ImageNet[4]上的神经网络，其目标是分类，所以我们无法确定fid对某些不自然的图像变形会有何反应，也无法说明fid用来评价一些与ImageNet领域无关任务（比如人脸生成和数字、字体生成）的合理性[1]。

### 1近邻分类器
1. 先用当前的G网络生成图片，然后从数据集中随机采样出一些图片。
2. 随后对每张图片存下对应的像素空间（输入），卷积空间（卷积层输出），类别分数（fc层输出），类别概率（softmax输出）
3. 对大小为n的mini_batch，计算出真实数据和生成数据之间和各自的n\*n距离矩阵，n\*n矩阵的计算方法为(x\*\*2 - 2x\*y + y\*\*2)
4. 根据距离进行1nn分类，即不算自己和自己的距离，看和自己距离最近的特征的标签是哪一个，以此计算一系列tp、fp等分类数值

## 人脸质量评价方法

# 参考
[1] Borji A. Pros and Cons of GAN Evaluation Measures[J]. arXiv preprint arXiv:1802.03446, 2018.
[2] Heusel M, Ramsauer H, Unterthiner T, et al. GANs trained by a two time-scale update rule converge to a Nash equilibrium[J]. arXiv preprint arXiv:1706.08500, 2017.
[3] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.
[4] Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical image database[C]//Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. Ieee, 2009: 248-255.