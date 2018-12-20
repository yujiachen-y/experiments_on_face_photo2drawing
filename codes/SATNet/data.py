"""
Modified from https://github.com/NVlabs/MUNIT/blob/master/data.py
"""
import json
import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class ImageLabelFileInfo(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([os.path.splitext(path)[0] for path in os.listdir(os.path.join(root, 'informations'))])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = []
        for idx, class_name in enumerate(self.classes):
            self.imgs += [(filename, idx) for filename in json.load(open(os.path.join(root, 'informations', class_name+'.json')))]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, 'images', impath+'.jpg'))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class WCDataset(data.Dataset):
    def __init__(self, dataset_path, is_train, data_type, clear_mode=False, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        training_file = os.path.join(
            dataset_path,
            'EvaluationProtocols',
            'FaceVerification',
            'UnRestricted',
            'UnRestrictedView1_Dev%s.txt' % ('Train' if is_train else 'Test'),
        )
        with open(training_file) as f:
            self.class_num = int(f.readline())
            self.class_names = []
            self.images = []
            for i in range(self.class_num):
                words = f.readline().split()
                class_name = ' '.join(words[:-2])
                self.class_names.append(class_name)
                if data_type == 'c':
                    file_string = 'C%05d'
                    img_num = int(words[-2])
                elif data_type == 'p':
                    file_string = 'P%05d'
                    img_num = int(words[-1])
                else:
                    assert 0, 'only support data_type in {c|p}'
                
                for j in range(img_num):
                    if clear_mode:
                        landmark = load_landmark(os.path.join(dataset_path, 'FacialPoints', class_name, file_string%(j+1)+'.txt'))
                        if not is_corrected_landmark(landmark):
                            continue

                    self.images += [(os.path.join(dataset_path, 'OriginalImages', class_name, file_string%(j+1)+'.jpg'), i)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        img = self.loader(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_landmark(landmark_file_path):
    return [
        tuple(map(float, landmark.strip().split(' '))) for landmark in open(landmark_file_path).readlines()
    ]


def check_landmark_position(landmark):
    landmark = np.array(landmark)

    # check 1, 2, 3, 4
    if not (landmark[0, 1] <= np.min(landmark[0:4, 1]) and\
            landmark[1, 0] <= np.min(landmark[0:4, 0]) and\
            landmark[2, 1] >= np.max(landmark[0:4, 1]) and\
            landmark[3, 0] >= np.max(landmark[0:4, 0])):
        return False

    # check 5, 6, 7, 8
    if not (landmark[4, 0] < landmark[5, 0] < landmark[6, 0] < landmark[7, 0]):
        return False

    # check 9, 10, 11, 12
    if not (landmark[8, 0] < landmark[9, 0] < landmark[10, 0] < landmark[11, 0]):
        return False

    # check 13, 14, 15, 16, 17
    if not (landmark[14, 0] < np.min((landmark[12, 0], landmark[13, 0], landmark[15, 0],)) <\
                              np.max((landmark[12, 0], landmark[13, 0], landmark[15, 0],)) <\
                              landmark[16, 0] and\
            landmark[12, 1] < np.min(landmark[13:17, 1])):
        return False

    # check horizontal
    if not (landmark[1, 0] < np.min(landmark[12:16, 0]) < landmark[3, 0]):
        return False

    # check vertical
    if not (landmark[0, 1] < np.min(landmark[4:8, 1]) < np.max(landmark[4:8, 1]) <\
                             np.min(landmark[8:12, 1]) < np.max(landmark[8:12, 1]) <\
                             np.min(landmark[12:16, 1]) < np.max(landmark[12:16, 1]) < landmark[2, 1]):
        return False
    
    return True


def is_corrected_landmark(landmark, offset=2, corrected_rate=0.10, resize_factor=2):
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

    if not check_landmark_position(landmark):
        return False

    ref_pts = [
        [30.2946+offset, 51.6963+offset],
        [65.5318+offset, 51.5014+offset],
        [48.0252+offset, 71.7366+offset],
        [33.5493+offset, 92.3655+offset],
        [62.7299+offset, 92.2041+offset],
    ]
    w, h = (96+offset*2, 112+offset*2)
    landmark = get_img5point(landmark)
    w, h = w * resize_factor, h * resize_factor
    for i, ld in enumerate(ref_pts):
        ref_pts[i] = [ld[0]*resize_factor, ld[1]*resize_factor]

    for ld0, ld1 in zip(ref_pts, landmark):
        if not (0 <= ld1[0] < w and 0 <= ld1[1] < h):
            return False
        ld0, ld1 = np.array(ld0), np.array(ld1)
        if np.sum((ld0 - ld1) ** 2) * np.pi > w * h * corrected_rate:
            return False

    return True


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
