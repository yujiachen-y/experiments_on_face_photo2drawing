import os

import cv2
import numpy as np
from tqdm import tqdm

# 这里注意x代表的是横轴，y代表的是纵轴
imgSize = np.array([128, 128])
coord5point = np.array([
    (30.2946, 51.6963,),        # Left eye
    (65.5318, 51.5014,),        # Right eye
    (48.0252, 71.7366,),        # Nose tip
    (33.5493, 92.3655,),        # Mouth left corner
    (62.7299, 92.2041,),        # Mouth right corner
]) + np.array([16, 8])

imgSize *= 4
coord5point *= 4

# change imgSize to tuple
imgSize = tuple(map(int, imgSize))

def warp_img(img, landmark):
    def calc_eye_point(landmark, is_right_eye=0):
        offset = is_right_eye * 2
        t = np.array([
            landmark[8+offset],
            landmark[9+offset],
        ])
        return t.mean(axis=0)
    
    img5point = np.array([
        calc_eye_point(landmark, is_right_eye=0),       # Left eye
        calc_eye_point(landmark, is_right_eye=1),       # Right eye
        landmark[12],                                   # Nose tip
        landmark[14],                                   # Mouth left corner
        landmark[16],                                   # Mouth right corner
    ])

    img = img.astype(np.uint8)
    M = cv2.estimateRigidTransform(img5point, coord5point, fullAffine=True)
    return cv2.warpAffine(img, M, imgSize)


def generate_dataset_face_frontalization():
    from . import config
    from .datas import get_dirs, get_image, get_overview
    from .utils import dataset_iterator, perpare_dataset_dir, im_str_to_np

    # get overview
    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(config.WC_original_dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)

    # perpare_dataset_dir
    new_dataset_names = ['frontalization_dataset',]
    new_dataset_dirs = []
    for new_dataset_name in new_dataset_names:
        new_dataset_dirs.append(perpare_dataset_dir(new_dataset_name, __file__))
    new_images_dir = os.path.join(new_dataset_dirs[0], config.WC_original_images_dir_name)

    # face_frontalization
    for people_name, image_type, image_name, landmark in tqdm(dataset_iterator(config.WC_original_dataset_name)):
        im_str = get_image(config.WC_original_dataset_name, people_name, image_name, show_landmark=0)
        im = im_str_to_np(im_str)
        im = warp_img(im, landmark)

        people_name = people_name.replace('_', ' ')
        file_dir = os.path.join(new_images_dir, people_name)
        os.makedirs(file_dir, exist_ok=True)
        os.chdir(file_dir)
        
        if not cv2.imwrite(image_name+'.jpg', im):
            assert 0, "cv2.imwrite failed"


if __name__ == '__main__':
    # from . import config
    # from .utils import dataset_iterator, get_image

    # def imshow(im):
    #     import matplotlib.pyplot as plt
    #     plt.imshow(im[:, :, ::-1])
    #     plt.show()

    # dataset_name = config.WC_original_dataset_name
    # for people_name, image_type, image_name, landmark in dataset_iterator(dataset_name):
    #     im_str = get_image(dataset_name, people_name, image_name, show_landmark=0)
    #     im = np.fromstring(im_str, np.uint8)
    #     im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    #     im = warp_img(im, landmark)
    #     imshow(im)
    generate_dataset_face_frontalization()