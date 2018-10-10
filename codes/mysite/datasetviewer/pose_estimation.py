import cv2
import numpy as np


def pose_estimation(im_str, landmark):
    # get cv2 image
    im = np.fromstring(im_str, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    size = im.shape

    # 2D image points.
    image_points = np.array([
        landmark[8],       # Left corner of left eye
        landmark[9],       # Right corner of left eye
        landmark[10],      # Left corner of right eye
        landmark[11],      # Right corner of right eye
        landmark[12],      # Nose tip
        landmark[2],       # Contour(Chin)
    ], dtype='double')

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
    model_points = np.array([
        (-40.0842667, 4.5685606, 71.7215538),       # Left corner of left eye
        (-4.4239935, 6.4678698, 83.1831589),        # Right corner of left eye
        (22.1054259, 4.9072615, 82.2120888),        # Left corner of right eye
        (60.6649881, 5.1719070, 66.5766238),        # Right corner of right eye
        (9.6113257, -35.8565550, 108.0227578),      # Nose tip
        (8.8716316, -100.8706606, 70.5823179),      # Contour(Chin)
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


def generate_side_face_dataset():
    import os, shutil, hashlib

    from . import config
    from .utils import get_dirs, get_overview, get_image

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
    for people_name in people_names:
        for image_type in ('c', 'p',):
            for image_name in image_names[people_name][image_type]:
                image = get_image(config.WC_original_dataset_name, people_name, image_name, show_landmark=0)
                euler_angle = pose_estimation(image, landmarks[people_name][image_name])
                print(euler_angle)
                import pdb; pdb.set_trace()


if __name__ == '__main__':
    generate_side_face_dataset()