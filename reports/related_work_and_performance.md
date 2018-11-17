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

# 数据集检测

## fid

在实验开始前针对预处理好的数据集计算了fid，对应值如下：

- 训练集中caricature和photo之间的fid为：$85.995$
- 测试集中caricature和photo之间的fid为：$100.054$
- 训练集和测试集中两个caricature之间的fid为：$33.896$
- 训练集和测试集中两个photo之间的fid为：$18.553$

## 1近邻分类器

自己担心数据集过小，会不会导致fid的表现有误差，根据[5]中的实验现象，fid在小数据上还是有一定偏差的，以及[2]中也建议fid的计算时至少需要10000张图片，不然fid值是估计不足的。

因此1近邻分类器在此实验中可以起到对照和增强实验严谨性的作用。

## 人脸识别评价

为了说明这项工作生成人脸的可识别程度，应该添加对应的人脸识别评价指标，来说明网络对人脸信息的保留能力。

不过这个工作有些复杂，首先是网络如何识别画像，是否需要fine-tuning，画像的数量是否够多来支持fine-tuning？其次还有很多实现上的细节可能会让我们花费太多的时间。所以为了尽快完成毕业设计，这一块先放着不做。

# 相关工作

本人的开题报告中已有相关工作的整体介绍，在此就不赘述。接下来详细介绍一下MUNIT和CycleGAN在此任务上的表现。

## MUNIT

### 实验设置

实验设置与论文[10]中的一致。

#### 数据预处理

1. 把图像变成大小为$(C, H, W)$，值为$[0.0, 1.0]$之间的张量
2. 对图像进行归一化，把3个channel的均值和方差设置为$(0.5, 0.5)$
3. 对图像进行随机的裁切，并缩放到指定的大小
4. 以$0.5$的概率对图像进行随机的水平翻转

#### 网络结构

一个卷积块内部操作的排布是：卷积 - 归一化 - 非线性。

##### 结构定义

- c7s1-k 代表一个$7 \times 7$的卷积块，这个卷积块有$k$个滤波器，步长为$1$。

- dk 代表一个$4 \times 4$的卷积块，这个卷积块有$k$个滤波器，步长为$2$。

- Rk 代表一个残差卷积块，这个残差卷积块中含有两个$3 \times 3$的卷积块，每个卷积块有$k$个滤波器，步长为$1$。

- uk 代表一个$2 \times 2$的最近邻上采样层和一个$5 \times 5$的卷积块，该卷积块有$k$个滤波器，步长为$1$。

- GAP 代表一个全局池化层，把输入的特征大小变为$(1, 1)$。

- fck 代表一个有$k$个滤波器的全连接层。

##### 生成器

- 非线性函数：ReLU
- padding方法：reflect padding
- 归一化方法：内容编码器使用了Instance Normalization，风格编码器没有使用归一化，解码器使用了Adaptive Instance Normalization

**具体结构**

- 内容编码器：c7s1-64, d128, d256, R256, R256, R256, R256
- 风格编码器：c7s1-64, d128, d256, d256, d256, GAP, fc8
- 解码器：R256, R256, R256, R256, u128, u64, c7s1-3

##### 判别器

- 非线性函数：Leaky ReLU，斜率为$0.2$
- padding方法：reflection padding
- 归一化方法：无

**具体结构**

d64, d128, d256, d512

使用了3个不同规模，目标函数是LSGAN的判别器。

#### 损失函数

##### 重构损失

$x_1, c_1, s_2$经过编码解码的过程后，保持不变。
$$
\mathcal{L}_{\rm{recon}}^{x_1} = \Bbb{E}_{x_1 \sim p(x_1)}[\left \| G_1(E^c_1(x_1), E^s_1(x_1))-x_1 \right \|_1] \tag 1
$$

$$
\begin{align}
\mathcal{L}_{\rm{recon}}^{c_1} = \Bbb{E}_{c_1 \sim p(c_1), s_2 \sim q(s_2)[\left \| E^c_2(G_2(c_1, s_2)) - c_1 \right \|_1]} \tag 2 \\
\mathcal{L}_{\rm{recon}}^{s_2} = \Bbb{E}_{c_1 \sim p(c_1), s_2 \sim q(s_2)[\left \| E^s_2(G_2(c_1, s_2)) - s_2 \right \|_1]} \tag 3 \\
\end{align}
$$

其中，$q(s_2)$来自正态分布$\mathcal{N}(0, \rm{I})$，$p(c_1)$中$c_1=E^c_1(x_1)$以及$x_1 \sim p(x_1)$。

##### 对抗损失

生成的图像符合真实图像的分布。要注意的是实验代码中实际使用的是LSGAN，因此公式(4)中的log操作符在实验中要换成均方误差。
$$
\mathcal{L}_{\rm{GAN}}^{x_2} = \Bbb{E}_{c1 \sim p(c_1), s_2 \sim q(s_2)[\mathrm{log}(1 - D_2(G_2(c_1, s_2)))]} + \Bbb{E}_{x_2 \sim p(x_2)}[\mathrm{log}D_2(x_2)] \tag 4
$$

##### 目标方程

$$
\min\limits_{E_1, E_2, G_1, G_2} \max\limits_{D_1, D_2} \mathcal{L}(E_1, E_2, G_1, G_2, D_1, D_2) = \mathcal{L}_\mathrm{GAN}^{x_1} + \mathcal{L}_\mathrm{GAN}^{x_2} + \\\lambda_x(\mathcal{L}_{\rm{recon}}^{x_1} + \mathcal{L}_{\rm{recon}}^{x_2}) + \lambda_c(\mathcal{L}_{\rm{recon}}^{c_1} + \mathcal{L}_{\rm{recon}}^{c_2}) + \lambda_s(\mathcal{L}_{\rm{recon}}^{s_1} + \mathcal{L}_{\rm{recon}}^{s_2}) \tag 5
$$

其中，$\lambda_x=10, \lambda_c=1, \lambda_s=1$

另外，MUNIT的实际实验代码还在目标方程中添加了$x_1$和$x_2$的循环一致性损失，如下：

##### 循环一致性损失

图像$x_1$转到其他domain再次转回原domain后，保持不变。
$$
\mathcal{L}_{\mathrm{cc}}^{x_1} = \Bbb{E}_{x_1 \sim p(x_1), s_2 \sim q(s_2)}[\left \| G_1(E_2^c(G_2(E_1^c(x_1), s_2)), E_1^s(x_1)) - x_1 \right \|_1] \tag 6
$$
循环一致性损失在目标方程中的系数是$\lambda_{cc}=10$

#### 超参数

- 优化算法：Adam $\beta_1=0.5, \beta_2=0.999$
- 学习率：初始值$0.0001$，每$100,000$次迭代学习率递减为原来的一半
- batch size：每个batch包含$1$个图像
- 迭代次数：$1,000,000$次，即遍历$1,000,000$个mini_batch后结束训练
- 模型初始化：```torch.nn.init.kaiming_normal_(data, a=0, mode='fan_in', nonlinearity='leaky_relu')```[13]
- style code：维度为$8$的一维向量

### 测试结果

目前正在训练中

## CycleGAN

### 实验设置

实验设置与论文[11, 12]中的源代码一致

#### 数据预处理

在CycleGAN实验中，一个图像会经过如下的预处理流程：

1. 把图像缩放到$(286, 286)$的大小
2. 对图像进行随机的裁切，得到大小为$(256, 256)$的图片
3. 以$0.5$的概率对图像进行随机的水平翻转
4. 把图像变成大小为$(C, H, W)$，值为$[0.0, 1.0]$之间的张量
5. 对图像进行归一化，把3个channel的均值和方差设置为$(0.5, 0.5)$

#### 网络结构

一个卷积块内部操作的排布是：卷积 - 归一化 - 非线性。

##### 结构定义

- c7s1-k 代表一个$7 \times 7$的卷积块，这个卷积块有$k$个滤波器，步长为$1$。

- dk 代表一个$3 \times 3$的卷积块，这个卷积块有$k$个滤波器，步长为$2$。

- Rk 代表一个残差卷积块，这个残差卷积块中含有两个$3 \times 3$的卷积块，每个卷积块有$k$个滤波器，步长为$1$。

- uk 代表一个$2 \times 2$的最近邻上采样层和一个$3 \times 3$的卷积块，该卷积块有$k$个滤波器，步长为$1$。
- Ck 代表一个$4 \times 4$的卷积块，这个卷积块有$k$个滤波器，步长为$2$。

##### 生成器

- 非线性函数：Leaky ReLU，斜率为$0.2$
- padding方法：reflection padding
- 归一化方法：Instance Normalization

**具体结构**

c7s1-32, d64, d128, R128, R128, R128, R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3

##### 判别器

- 非线性函数：Leaky ReLU，斜率为$0.2$
- padding方法：zero padding
- 归一化方法：Instance Normalization

使用了$70 \times 70$的PatchGANs[12]，并把目标函数设置为LSGAN。

为了避免模型震荡[14]，判别器的更新采用了[15]中的策略：使用以前的生成图片作为判别器的输入，而不是现在的生成图像。具体在代码中的实现是：设置一个图片池，如果池的大小不到$50$，就把当前生成的图片存在图片池中，并把当前生成的图片作为判别器的输入；如果池的大小超过$50$，就用$0.5$的概率，决定要不要把当前的生成图片和池中的某一个图片交换，然后再作为判别器的输入。

**具体结构**

C64, C128, C256, C512

注意：C64后面没有跟着任何归一化层。

#### 损失函数

##### 对抗损失

生成的图像符合真实图像的分布。要注意的是实验代码中实际使用的是LSGAN，因此公式(4)中的log操作符在实验中要换成均方误差。
$$
\mathcal{L}_\text{GAN}(G, D_Y, X, Y) = \Bbb{E}_{y \sim p_{\text{data}(y)}}[\log D_Y(y)] +\Bbb{E}_{x \sim p_{\text{data}(x)}}[\log (1 - D_Y(G(x)))] \tag 7
$$

##### 循环一致性损失

图像$x$和$y$转到其他domain再次转回原domain后，保持不变。
$$
\mathcal{L}_\text{cyc}(G, F) = \Bbb{E}_{x \sim p_{\text{data}(x)}}[\left \| F(G(x)) - x \right \|_1] +\Bbb{E}_{y \sim p_{\text{data}(y)}}[\left \| G(F(y)) - y \right \|_1] \tag 8
$$

##### 目标方程

$$
\arg\min_\limits{G,F}\max_\limits{D_x, D_y} \mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_\text{GAN}(G, D_Y, X, Y) + L_\text{GAN}(F, D_X, Y, X) + \lambda\mathcal{L}_\text{cyc}(G, F) \tag 9
$$

其中$\lambda=10$

另外，CycleGAN的实验代码中还添加了身份损失，用来保证图片中的信息在转换后不会发生改变：

##### 身份损失

图像$x$和$y$转到其他domain后得到的图像，要和转换前的图像相似。
$$
\mathcal{L}_\text{idt}(G, F) = \Bbb{E}_{x \sim p_{\text{data}(x)}}[\left \| G(x) - x \right \|_1] + \Bbb{E}_{y \sim p_{\text{data}(y)}}[\left \| F(y) - y \right \|_1]
$$
因为身份损失和人脸照片-画像之间的转化不符合，所以自己在实验复现中设置其权重为$\lambda_{idt}=0$

#### 超参数

- 优化算法：Adam $\beta_1=0.5, \beta_2=0.999$
- 学习率：初始值$0.0002$，训练到第$100$个epoch后，学习率开始线性递减，直到epoch$200$时，训练结束，学习率变为$0$
- batch size：每个batch包含$1$个图像
- 迭代次数：$200$ epoch，遍历完$200$次dataset后结束训练
- 模型初始化：用正态分布$\mathcal{N}(\text{0}, \text{0.02})$来初始化模型，```torch.nn.init.normal_(data, mean=0, std=0.02)```[13]

### 测试结果

目前正在训练中

# 未来工作

- [ ] 添加1近邻分类器的分类结果作为参照，增强实验的可信程度。（自己已做的实验都存下了每个阶段的模型和对应的测试集输出，所以这个做一做还是很快的，但是自己赶着完成毕设，这一段就不做了）
- [ ] 对工作生成的人脸进行人脸识别的相关指标评价，说明网络对人脸信息的保留能力。

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

[10] Huang X, Liu M Y, Belongie S, et al. Multimodal Unsupervised Image-to-Image Translation[J]. arXiv preprint arXiv:1804.04732, 2018.

[11] Zhu J Y, Park T, Isola P, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks[J]. arXiv preprint, 2017.

[12] Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[J]. arXiv preprint, 2017.

[13] He K, Zhang X, Ren S, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1026-1034.

[14] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]//Advances in neural information processing systems. 2014: 2672-2680.

[15] Shrivastava A, Pfister T, Tuzel O, et al. Learning from Simulated and Unsupervised Images through Adversarial Training[C]//CVPR. 2017, 2(4): 5.