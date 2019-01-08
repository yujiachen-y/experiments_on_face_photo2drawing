[中文README](README-zh.md)

This repository is my personal bachelor's graduation project, which is a variety of **unorganized dirty code**. In this README, first I will introduce the contents of my graduation project, and then explain the structure of the folders in this repo.

**Say something nonsense, I implemented the pytorch SAGAN in this repository([code](codes/SATNet/networks.py#L311)|[paper](https://arxiv.org/abs/1805.08318)), although the actual operation is not much different from other implementation versions of the current online, but I am quite satisfied with the effect of SAGAN, so just mark.**

**I also tried to implement Deformable ConvNets v2([paper](https://arxiv.org/abs/1811.11168)), but the implemented code has a bug that cannot load pre-training weights. I guess that the problem may related to the operation on the learning rate of the Deformable module. Because I have no good solution at present, besides the graduation project is over, so I deleted the branch which have the code of Deformable module.**

# Introduction to the content of my graduation project
This study hopes to realize the mutual conversion of face photos and drawings. At the begin, I hope it doesn't like stylized that does not change the image content, and changes the texture only. The drawings generated according to the face photos are "artistic", retaining the identity of the person's face, but with exaggerated details. At the same time, the face photos generated from the face drawings can also retain the identity features displayed in the drawings, and have the same face structure as the real world have.

The current results and goals still have a big gap. The main shortcomings are two: First, in terms of photo generation drawing, the model has mode dropping, and the currently generated drawings are very similar to the stylized results. I think the reason is that the style of the drawings in the dataset is too scattered. If you consider only the transformation of a certain style and collect enough data, you should be able to alleviate this problem. Secondly, in terms of photo generation, the similarity is still satisfactory, but the similarity in identity is not particularly good. Under the supervision of the pre-trained and fine-tuned SphereFace network, the current network can preserve the identity of the drawing to some extent, but some photos generated from male drawings will be painted with lipstick or dermabrasion, which looks very similar to women. In this respect, I theink we can consider strengthening the supervision of SphereFace. This part of the experiment did not do it because of I haven't enough time.

Here is the current result
![results1](reports/elements/SATNet_c2p_results.jpg)

## Introduction to the face recognition part
In order to better preserve the identity features in the image translation results, this project uses a face recognition network to guide the training network.

This project uses pre-training weights from [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch), and fine-tuned it for [WebCaricature dataset](https://cs.nju.edu.cn/rl/WebCaricature.htm). The result are better then [the WebCaricature paper](https://arxiv.org/abs/1703.03230), see the following table:

| method | FAR=0.1%(%) | FAR=1%(%) | AUC |
|------|-------------|-----------|-----|
| SIFT-Land | 4.43±0.82 | 15.24±2.03 | 0.780±0.017 |
| VGG-BOX | 32.07±2.60 | 56.76±2.35 | 0.946±0.005 |
| VGG-Eye | 19.24±1.95 | 40.88±2.23 | 0.898±0.007 |
| **SphereFace-fine-tuned** | **94.58±3.99** | **96.82±2.50** | **0.996±0.004** |

[fine-tuned SphereFace weights](https://drive.google.com/open?id=1esOigCk0lCPM8ZE3dSS3zJ_5oJ4wTq0Y)

The current loss of face recognition is a simple form of hinge, ie L = max(0, threshold - dist). I think that I can try to refer to the Knowledge Distillation in the next step, that is, the generated image should be consistent with a certain feature of the original image in the high-dimensional space, and the intensity of the learning is controlled by a parameter similar to T.

## Introduction to the dataset preprocessing
In order to cooperate with the face recognition network to supervise the training, I corrected the [WebCaricature dataset](https://cs.nju.edu.cn/rl/WebCaricature.htm) and removed the side face image after the correction and the image with the deviation between the facial features and the reference key points. See the [code](codes/SATNet/data.py#L257).

The current results show that the training results of the model will perform better after the non-standard data is eliminated.

## Introduction to the image translation part
Talk about the main model.

The main research content of this project is about the mutual conversion of face photos and drawings. It can be regarded as a special image translation problem.(If the setting of face in network traning is removed, the network training is removed, the network model in the project can be used for general image translation problem.)

The project adopts a method of stacking various new technologies to improve the evaluation results of training results in the network structure and training strategy. The newly added new technologies include adversarial loss in the form of hinge([paper](https://arxiv.org/abs/1702.08896)), SAGAN([code](codes/SATNet/networks.py#L311)|[paper](https://arxiv.org/abs/1805.08318)), Spectral Normalization([paper](https://arxiv.org/abs/1802.05957)) and TTUR([paper](https://arxiv.org/abs/1706.08500))).

In order to save time, the main traning code is copied from MUNIT([code](https://github.com/NVlabs/MUNIT)|[paper](https://arxiv.org/abs/1804.04732)), and then it is modified and added by myself, and then used for training.

The basic assumptions of this project are also from MUNIT([code](https://github.com/NVlabs/MUNIT)|[paper](https://arxiv.org/abs/1804.04732)), which can be considered as a special enhancement of MUNIT's task of transforming face photos and drawings.

# Folder structure
**The code I wrote was messy and unorganized. If anyone wants to do some improved experiments based on my code, I suggest reorganizing the code yourself, otherwise the code management will be very bad.**
```
--codes                 // Code folder
 |--3rdparty            // 3rd party software library
 |--dltools             // Help training code
 |--mysite              // Django folder
   |--datasetviewer     // Django app to display datasets
   |--mysite            // Django project setup code
   |--stylized_face     // Online stylized image of django app
 |--recoder             // useless, ignore it
 |--SATNet              // Self-attention-transfer-net, the name is obtained casually
 |--sphereface          // Sphereface fine tunning code
--reports               // Some reports written by myself, the content is a bit messy and embarrassing
```

# Run command
Please note that some commands are with parameters, the function of the parameters can be found in the corresponding code.
```
// Training SATNet
python codes/dltools/fid_trainer.py
// Start the django project
python codes/mysite/manage.py runserver
// face correction on the dataset
python codes/mysite/warp.py
// sphereface fine-tunning
python codes/sphereface/transfer_learning.py
```
## Requirements
If you find that some dependencies are not in the list below, please rule me out.
- python 3.6.5
- pytorch
- numpy
- cv2
- django
- dlib

# Experimental results
- Eliminating side-face images in the dataset can lead to better experimental results
- The self-attention module is conducive to the reduction of FID, but in the case of small resolution (96*112), the result of the model generated by the self-attention module tends to have a checkerboard effect.
- LSGAN has a lot of restrictions on the model, and the result is a bit single. The GAN in the form of hinge will not have this problem.

# Future work
Talk about the points that I think are worth trying, but I have not had time to do it.
- Want to try and use deformable convnet to help generate better results
- Try the knowledge distillation. At present, the role of SphereFace in training is just to make a identity loss. I think that there is still a lot of room for experimentation on how to use the face recognition network.

## Supervise learning direction
You can also consider designing training in the direction of supervised learning. Each image in the [WebCaricature dataset](https://cs.nju.edu.cn/rl/WebCaricature.htm) has identity information. If you consider forcing a loss between the generated photo (or drawing) of the same identity and the real photo (or drawing), will it imporve network performance?

## Unsupervised learning direction
Designing traning in an unsupervised learning direction can reduce the reliance on specific datasets. And I actually thought about whether I could use a large photo dataset with identity tags and another large drawing dataset with identity tags. Specifying the specific style of the drawing should be able to train a model with good results. However, there are still some problems in considering the landmark extraction work of the drawing. The input format cannot be fixed, which will cause the module in the model to be globally unawaer. The failure of the global module learning means that the model cannot perceive the global attribute of the input ( That is the overall characteristics of the face).