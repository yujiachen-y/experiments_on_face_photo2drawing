<center><h1>毕业设计网络结构和实验记录</h1></center>

<center><i>written by  <a href="http://yujiachen.top/">Yu, Jiachen</a></i></center>

# 设计网络考虑的各要素
## 数据集
理想的情况是采用没有经过处理的原始数据集，直接作为网络的输入。但这样最大的问题是如何验证网络保留身份特征的能力。一般的神经网络对没有经过转正的人脸的信息提取能力是很差的，虽然我们可以考虑通过送入landmark的形式来帮助网路提取人脸信息，但是要考虑到因为loss中要上一个一致性loss，所以不可避免地有一个对生成的图像进行再解码的过程，但是**目前还没有足够好的方法可以生成画像的landmark**。因此，对于生成的画像图片，网络是不能进行再解码的，这就导致了loss无法计算，这一点对网络的训练是致命的。

因此实际中还是采用被转正后的数据集，加上简单的数据清洗，进行训练和测试。

画像的landmark生成是一个大的课题，以及如何把landmark与画像生成结合起来也是一个尚未有人涉足的问题。因此本研究决定暂时不把这些还未解决的课题引入。

另：[1]中训练了一个landmark识别网络来把人脸照片的身份信息和姿势信息（landmark位置）分离开，对上述问题的研究应该会有启发。

## 训练方法
发现自己的训练代码跑起来总是比其他人的代码要慢，主要原因是batch数和迭代次数，batch数受限于网络结构和大小，过短的迭代次数又容易导致模型的性能下降，打算代码写完后进行几次测试，尽量让实验可以在一天到两天之内跑完。

# 基础结构
## Spectral Normalization
公式和相关的说明在下一次提交的时候补上。
Spectral Normalization[2]是一个利用参数的二次范式，根据Lipschitz限制推导出的一个参数优化方法，其主要目的是为了让辨别器在训练过程中稳定。
另外[2]的作者还提出了一个带投影的CGAN辨别器[5]，

## Deformable Convolutional Networks
公式和相关的说明在下一次提交的时候补上。
Deformable Convolutional Networks[3]会根据输入在卷积前进行不规则的采样和插值计算，以便让不同位置的卷积有着不同的感受域。个人认为Deformable Conv可以认为是SAGAN的一种特殊情况，具体说明可见SAGAN的说明。因此本实验不采用Deformable Conv。

## Self-Attention Generative Adversarial Networks
公式和相关的说明在下一次提交的时候补上。
关于如何将SAGAN的公式转换为Deformable Conv，也在下一次提交的时候补上。

# 网络结构

# 参考
[1] Xie W, Shen L, Zisserman A. Comparator networks[J]. arXiv preprint arXiv:1807.11440, 2018.

[2] Miyato T, Kataoka T, Koyama M, et al. Spectral normalization for generative adversarial networks[J]. arXiv preprint arXiv:1802.05957, 2018.

[3] Dai J, Qi H, Xiong Y, et al. Deformable convolutional networks[J]. CoRR, abs/1703.06211, 2017, 1(2): 3.

[4] Zhang H, Goodfellow I, Metaxas D, et al. Self-Attention Generative Adversarial Networks[J]. arXiv preprint arXiv:1805.08318, 2018.

[5] Miyato T, Koyama M. cGANs with projection discriminator[J]. arXiv preprint arXiv:1802.05637, 2018.