import cv2
import numpy as np

from tqdm import tqdm


def pose_estimation(im_str, landmark, type):
    # get cv2 image
    im = np.fromstring(im_str, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    size = im.shape

    # 2D image points.
    if type == 0:
        image_points = np.array([
            landmark[8],        # Left corner of left eye
            landmark[9],        # Right corner of left eye
            landmark[10],       # Left corner of right eye
            landmark[11],       # Right corner of right eye
            landmark[12],       # Nose tip
            landmark[2],        # Contour(Chin)
            landmark[13],       # Mouth upper lip top
            landmark[14],       # Mouth left corner
            landmark[15],       # Mouth lower lip bottom
            landmark[16],       # Mouth right corner
        ], dtype='double')
    elif type == 1:
        image_points = np.array([
            landmark[12],       # Nose tip
            landmark[2],        # Contour(Chin)
            landmark[8],        # Left corner of left eye
            landmark[11],       # Right corner of right eye
            landmark[14],       # Mouth left corner
            landmark[16],       # Mouth right corner
        ], dtype='double')
    else:
        assert False, 'unknow type: {}'.format(type)

    # 3D model points.
    # This 3D model comes from
    # T. Bolkart, S. Wuhrer
    # 3D Faces in Motion: Fully Automatic Registration and Statistical Analysis
    # Computer Vision and Image Understanding, 131:100–115, 2015
    # and
    # T. Bolkart, S. Wuhrer
    # Statistical Analysis of 3D Faces in Motion
    # 3D Vision, 2013, pages 103-110
    #
    # http://facepage.gforge.inria.fr/Downloads/multilinear_face_model.7z
    if type == 0:
        model_points = np.array([
            (-40.0842667, 4.5685606, 71.7215538),       # Left corner of left eye
            (-4.4239935, 6.4678698, 83.1831589),        # Right corner of left eye
            (22.1054259, 4.9072615, 82.2120888),        # Left corner of right eye
            (60.6649881, 5.1719070, 66.5766238),        # Right corner of right eye
            (9.6113257, -35.8565550, 108.0227578),      # Nose tip
            (8.8716316, -100.8706606, 70.5823179),      # Contour(Chin)
            (7.8067696, -56.3137785, 90.8181463),       # Mouth upper lip top
            (-17.9012600, -62.3345513, 72.0136693),     # Mouth left corner
            (8.6057675, -73.9771469, 84.0426881),       # Mouth lower lip bottom
            (37.6207788, -61.9098011, 70.2096017),      # Mouth right corner
        ], dtype='double')
    elif type == 1:
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Contour(Chin)
            (-225.0, 170.0, -135.0),    # Left corner of left eye
            (225.0, 170.0, -135.0),     # Right corner of right eye
            (-150.0, -150.0, -125.0),   # Mouth left corner
            (150.0, -150.0, -125.0),    # Mouth right corner
        ], dtype='double')

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        (focal_length, 0, center[0]),
        (0, focal_length, center[1]),
        (0, 0, 1),
    ], dtype='double')

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    # solvePnP
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Rotation Vector to Quaternion
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    w = np.cos(theta / 2)
    x = np.sin(theta / 2) * rotation_vector[0] / theta
    y = np.sin(theta / 2) * rotation_vector[1] / theta
    z = np.sin(theta / 2) * rotation_vector[2] / theta

    # Quaternoin to Euler Angle
    t0 = 2. * (w * x + y * z)
    t1 = 1. - 2. * (x * x + y * y)
    pitch = float(np.arctan2(t0, t1))

    t2 = 2. * (w * y - z * x)
    t2 = 1. if t2 > 1. else t2
    t2 = -1. if t2 < -1. else t2
    yaw = float(np.arcsin(t2))

    t3 = 2. * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    roll = float(np.arctan2(t3, t4))

    return pitch, yaw, roll


def calc_sloped_angle(pitch, yaw):
    # import ipdb; ipdb.set_trace()
    t = np.sqrt(np.sin(pitch) ** 2 + np.tan(yaw) ** 2)
    return np.arctan2(t, np.cos(pitch))


def generate_side_face_dataset():
    import os, shutil, hashlib
    
    from . import config
    from .utils import get_dirs, get_image, get_overview, dataset_iterator

    # get overview
    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(config.WC_original_dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)

    # makedir and backup
    new_dataset_name = 'side_face_dataset'
    version = sum(map(lambda x: x.startswith(new_dataset_name), os.listdir(config.WC_datasets_dir)))
    new_dataset_dir = os.path.join(config.WC_datasets_dir, 'side_face_dataset_v%03d' % version)
    os.makedirs(new_dataset_dir)
    shutil.copy(__file__, new_dataset_dir)

    # md5 check
    m = hashlib.md5()
    with open(__file__, 'rb') as f:
        m.update(f.read())
    md5_path = os.path.join(new_dataset_dir, 'md5')
    if not os.path.exists(md5_path):
        with open(md5_path, 'wb') as md5_file:
            md5_file.write(m.digest())
    else:
        with open(md5_path, 'rb') as md5_file:
            assert md5_file.read() == m.digest(), 'You can not change file: %s to override existing dataset!' % __file__

    # estimating pose
    new_filenames_dir = os.path.join(new_dataset_dir, config.WC_filenames_dir_name)
    for people_name, image_type, image_name, landmark in tqdm(dataset_iterator(config.WC_original_dataset_name)):
        im_str = get_image(config.WC_original_dataset_name, people_name, image_name, show_landmark=0)
        pitch, yaw, roll = pose_estimation(im_str, landmark, type=0)
        theta = np.abs(yaw) / np.pi * 180
        if theta > 30:
            people_name = people_name.replace('_', ' ')
            file_dir = os.path.join(new_filenames_dir, people_name)
            os.makedirs(file_dir, exist_ok=True)
            file_path = os.path.join(file_dir, config.WC_c_filename if image_type == 'c' else config.WC_p_filename)
            with open(file_path, 'a') as file:
                file.write(image_name+'.jpg\n')


if __name__ == '__main__':
    # import math

    # from . import config
    # from .utils import get_image, dataset_iterator

    # for people_name, image_type, image_name, landmark in dataset_iterator(config.WC_original_dataset_name):
    #     im_str = get_image(config.WC_original_dataset_name, people_name, image_name, show_landmark=0)
    #     pose0 = pose_estimation(im_str, landmark, type=0)
    #     pose1 = pose_estimation(im_str, landmark, type=1)
    #     print(people_name, image_name, '\n',
    #           pose0, pose0[1] / math.pi * 180, '\n',
    #           pose1, pose1[1] / math.pi * 180, '\n',)
    #     input()
    generate_side_face_dataset()