'''
copied and modified from: https://github.com/clcarwin/sphereface_pytorch/blob/master/lfw_eval.py
'''
from __future__ import print_function

import argparse
import bisect
import datetime
import os
import pickle
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from matlab_cp2tform import get_similarity_transform_for_cv2
from net_sphere import sphere20a

torch.backends.cudnn.bencmark = True


def calc_eye_point(landmark, is_right_eye=0):
        offset = is_right_eye * 2
        t = np.array([
            landmark[8+offset],
            landmark[9+offset],
        ])
        return t.mean(axis=0)


def get_img5point(landmark):
    return np.array([
        calc_eye_point(landmark, is_right_eye=0),       # Left eye
        calc_eye_point(landmark, is_right_eye=1),       # Right eye
        landmark[12],                                   # Nose tip
        landmark[14],                                   # Mouth left corner
        landmark[16],                                   # Mouth right corner
    ])


def load_landmark(file_path):
    return get_img5point([
        tuple(map(float, landmark.strip().split(' ')))\
        for landmark in\
        open(file_path).readlines()
    ])


def alignment(src_img,src_pts):
    ref_pts = [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(folds_length):
    folds = []
    n = sum(folds_length)
    n_folds = len(folds_length)
    l, r = 0, 0
    base = list(range(n))
    for i in range(n_folds):
        r += folds_length[i]
        test = base[l : r]
        train = list(set(base)-set(test))
        folds.append([train,test])
        l += folds_length[i]
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def calc_roc(predicts):
    def get_tar(thd, pos_predicts):
        return 1 - bisect.bisect_left(pos_predicts, thd) / len(pos_predicts)
    pos_predicts, neg_predicts = [], []
    for predict in predicts:
        if int(predict[1]) == 1:
            pos_predicts.append(float(predict[0]))
        else:
            neg_predicts.append(float(predict[0]))
    pos_predicts.sort()
    neg_predicts.sort()

    far3_idx = int(len(neg_predicts)-1-len(neg_predicts)*1e-3)
    far3 = get_tar(neg_predicts[far3_idx], pos_predicts)
    far2_idx = int(len(neg_predicts)-1-len(neg_predicts)*1e-2)
    far2 = get_tar(neg_predicts[far2_idx], pos_predicts)
    
    auc = 0
    n = len(pos_predicts) - 1
    for threshold in reversed(neg_predicts):
        while 0 <= n and threshold <= pos_predicts[n]:
            n -= 1
        auc += len(pos_predicts) - 1 - n
    auc /= len(pos_predicts) * len(neg_predicts)
    return far3, far2, auc


def get_predicts(dataset_path, model_path):
    model_name = os.path.splitext(os.path.split(model_path)[1])[0]
    feats_path = os.path.join(dataset_path, model_name+'.pkl')
    if os.path.exists(feats_path):
        predicts, folds_length = pickle.load(open(feats_path, 'rb'))
        return predicts, folds_length
    
    predicts=[]
    net = sphere20a()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    net.feature = True

    with open(os.path.join(dataset_path,
                           'EvaluationProtocols',
                           'FaceVerification',
                           'Restricted',
                           'RestrictedView2.txt',
                           )) as f:
        pairs_lines = iter(f.readlines())

    folds_length = []
    for fold_idx in range(10):
        fold_length = int(next(pairs_lines)) * 2
        folds_length.append(fold_length)
        for i in range(fold_length):
            p = next(pairs_lines)
            p = p.replace('\n', '').split('\t')
            if i * 2 < fold_length:
                sameflag = 1
                name = ' '.join(p[:-2])
                name1 = os.path.join(name, p[-2])
                name2 = os.path.join(name, p[-1])
            else:
                sameflag = 0
                name1, name2 = [], []
                for word in p:
                    if type(name1) != str:
                        if '00' in word:
                            name1 = os.path.join(' '.join(name1), word)
                        else:
                            name1.append(word)
                    else:
                        if '00' in word:
                            name2 = os.path.join(' '.join(name2), word)
                        else:
                            name2.append(word)
        
            img1 = os.path.join(dataset_path, 'OriginalImages', name1+'.jpg')
            landmark1 = load_landmark(os.path.join(dataset_path, 'FacialPoints', name1+'.txt'))
            img1 = alignment(cv2.imread(img1, 1), landmark1)
            img2 = os.path.join(dataset_path, 'OriginalImages', name2+'.jpg')
            landmark2 = load_landmark(os.path.join(dataset_path, 'FacialPoints', name2+'.txt'))
            img2 = alignment(cv2.imread(img2, 1), landmark2)

            imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
            for i in range(len(imglist)):
                imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
                imglist[i] = (imglist[i] - 127.5) / 128.

            img = np.vstack(imglist)
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
                output = net(img)
            f = output.data
            f1, f2 = f[0], f[2]
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append((cosdistance, sameflag))
    predicts = np.array(predicts)
    pickle.dump((predicts, folds_length), open(feats_path, 'wb'))
    return predicts, folds_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch sphereface wc')
    parser.add_argument('--wc', default='datasets/WebCaricature/original_dataset', type=str)
    parser.add_argument('--model','-m', default='codes/sphereface/sphere20a.pth', type=str)
    args = parser.parse_args()

    predicts, folds_length = get_predicts(args.wc, args.model)

    accuracy = []
    thd = []
    far3 = []
    far2 = []
    auc = []
    folds = KFold(folds_length)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        roc = calc_roc(predicts[test])
        far3.append(roc[0])
        far2.append(roc[1])
        auc.append(roc[2])
    print('WCACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    print('WCFAR3={:.4f} std={:.4f}'.format(np.mean(far3), np.std(far3)))
    print('WCFAR2={:.4f} std={:.4f}'.format(np.mean(far2), np.std(far2)))
    print('WCAUC={:.4f} std={:.4f}'.format(np.mean(auc), np.std(auc)))
