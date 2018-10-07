import os, cv2, numpy as np

def imshow(name, im):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

DATASET_PATH = 'D:\\File\\WebCaricature\\OriginalDataset'
FILE_C, FILE_P = 'file_c.txt', 'file_p.txt'

# prepare varable
images_dir = os.path.join(DATASET_PATH, 'OriginalImages')
landmarks_dir = os.path.join(DATASET_PATH, 'FacialPoints')
filenames_dir = os.path.join(DATASET_PATH, 'Filenames')

names = os.listdir(images_dir)

file_c = [s.strip()[:-4] for s in open(os.path.join(filenames_dir, names[0], FILE_C)).readlines()]
file_p = [s.strip()[:-4] for s in open(os.path.join(filenames_dir, names[0], FILE_P)).readlines()]

# # processed image
im_c = cv2.imread(os.path.join(images_dir, names[0], file_c[0]+'.jpg'))
# imshow('im_c', im_c)

im_p = cv2.imread(os.path.join(images_dir, names[0], file_p[0]+'.jpg'))
# imshow('im_p', im_p)

# add landmarks
lds_c = [tuple(map(int, map(float, s.strip().split(' ')))) for s in open(os.path.join(landmarks_dir, names[0], file_c[0]+'.txt')).readlines()]
im_ld_c = im_c.copy()
for ld in lds_c:
    x, y = ld
    im_ld_c[y-5:y+5, x-5:x+5, :] = np.array([0, 0, 0])
imshow('im_ld_c', im_ld_c)

lds_p = [tuple(map(int, map(float, s.strip().split(' ')))) for s in open(os.path.join(landmarks_dir, names[0], file_p[0]+'.txt')).readlines()]
im_ld_p = im_p.copy()
for ld in lds_p:
    x, y = ld
    im_ld_p[y-5:y+5, x-5:x+5, :] = np.array([0, 0, 0])
imshow('im_ld_p', im_ld_p)