import os

import cv2
import numpy as np
import torch
from skimage import io
from torchvision import transforms
from torchvision import utils as vutils

import dlib

try:
    from ...SATNet.data import default_loader
    from ...SATNet.networks import AdaINGen
    from ...sphereface.matlab_cp2tform import get_similarity_transform_for_cv2
except (ImportError, ValueError,):
    import sys
    sys.path.append(os.path.join(os.path.abspath('./codes'), 'SATNet'))
    sys.path.append(os.path.join(os.path.abspath('./codes'), 'sphereface'))
    from data import default_loader
    from networks import AdaINGen
    from matlab_cp2tform import get_similarity_transform_for_cv2

predictor_path = 'support_material/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


model_path = 'support_material/gen_00450000.pt'
config = {
    'dim': 64,
    'mlp_dim': 256,
    'style_dim': 8,
    'activ': 'relu',
    'n_downsample': 2,
    'n_res': 4,
    'pad_type': 'reflect',
}
gen_a = AdaINGen(3, config).cuda().eval()
gen_b = AdaINGen(3, config).cuda().eval()
gen_a.load_state_dict(torch.load(model_path)['a'])
gen_b.load_state_dict(torch.load(model_path)['b'])


transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5),
                                       (0.5, 0.5, 0.5))]
transform_list = [transforms.RandomCrop((224, 192))] + transform_list
transform_list = [transforms.Resize(200)] + transform_list
transform = transforms.Compose(transform_list)


def calc_eye_point(landmark, is_right_eye=0):
        offset = is_right_eye * 6
        t = np.array([
            landmark[36+offset],
            landmark[39+offset],
        ])
        return t.mean(axis=0)


def get_img5point(landmark):
    return np.array([
        calc_eye_point(landmark, is_right_eye=0),       # Left eye
        calc_eye_point(landmark, is_right_eye=1),       # Right eye
        landmark[30],                                   # Nose tip
        landmark[60],                                   # Mouth left corner
        landmark[64],                                   # Mouth right corner
    ])


def get_landmark(img_path):
    img = io.imread(img_path)
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = predictor(img, dets[0])
    landmark = [[p.x, p.y] for p in shape.parts()]
    return get_img5point(landmark)


def alignment(src_img_path, resize_factor=2):
    offset = 2
    ref_pts = [
        [30.2946+offset, 51.6963+offset],
        [65.5318+offset, 51.5014+offset],
        [48.0252+offset, 71.7366+offset],
        [33.5493+offset, 92.3655+offset],
        [62.7299+offset, 92.2041+offset],
    ]
    crop_size = (96+offset*2, 112+offset*2)
    src_pts = get_landmark(src_img_path)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32) * resize_factor
    crop_size = (crop_size[0]*resize_factor, crop_size[1]*resize_factor)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(cv2.imread(src_img_path), tfm, crop_size)

    dst_name = os.path.splitext(src_img_path)
    dst_img = dst_name[0]+'_alige'+dst_name[1]
    cv2.imwrite(dst_img, face_img)

    return dst_img


def stylize(img_path, seed=1):
    img = transform(default_loader(img_path)).unsqueeze(0).cuda()
    
    rng = np.random.RandomState(seed)
    s = torch.tensor(rng.randn(1, config['style_dim'], 1, 1), dtype=torch.float32).cuda()
    with torch.no_grad():
        c = gen_b.enc_content(img)
        img = gen_a.decode(c, s)

    img_name = os.path.splitext(img_path)
    out_path = img_name[0]+'_s%05d'%seed+img_name[1]
    vutils.save_image(img, out_path)
    return out_path


if __name__ == "__main__":
    from IPython import embed; embed()
