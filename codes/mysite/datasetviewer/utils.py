import os
from functools import lru_cache
from tqdm import tqdm

try:
    from . import config
except ImportError:
    import config


def get_dirs(dataset_name):
    '''
    give a dataset_name, and return
    (images_dir, filenames_dir, landmarks_dir),
    (fd_message, ld_message)
    or raise an Exception
    '''
    def get_dir(dataset_name, dir_name):
        result_dir = os.path.join(config.WC_datasets_dir,\
            dataset_name, dir_name)
        result_message = None
        if not os.path.exists(result_dir):
            tmp_result_dir = os.path.join(config.WC_datasets_dir,\
                config.WC_original_dataset_name, dir_name)
            result_message = "path: %s doesn't exits, so we use path: %s instead." % (result_dir, tmp_result_dir)
            result_dir = tmp_result_dir
            if not os.path.exists(tmp_result_dir):
                result_message = "path: %s doesn't exist" % tmp_result_dir
                result_dir = None
        return result_dir, result_message

    images_dir, id_message = get_dir(dataset_name, config.WC_original_images_dir_name)
    filenames_dir, fd_message = get_dir(dataset_name, config.WC_filenames_dir_name)
    landmarks_dir, ld_message = get_dir(dataset_name, config.WC_landmarks_dir_name)

    dirs = (images_dir, filenames_dir, landmarks_dir)
    messages = (id_message, fd_message, ld_message)

    for dir, message in zip(dirs, messages):
        if dir is None:
            raise config.DirNotFindError(message)

    return dirs, messages


@lru_cache(maxsize=config.cache_size)
def get_overview(images_dir, filenames_dir, landmarks_dir):
    '''
    params: images_dir, filenames_dir, landmarks_dir
    return: people_names: [people_name],
            image_names: {people_name: {'c' and 'p': filename}},
            landmarks: {filename: landmark}
    '''
    def get_filenames(filenames_dir, people_name):
        '''
        get a people's filenames from a filenames_dir
        return a dict which keys are ('c', 'p',) and values are list of image number like 'C00001'
        '''
        people_name = people_name.replace('_', ' ')
        return {
            'c': [os.path.splitext(filename.strip())[0]\
                for filename in\
                open(os.path.join(filenames_dir, people_name, config.WC_c_filename)).readlines()],
            'p': [os.path.splitext(filename.strip())[0]\
                for filename in\
                open(os.path.join(filenames_dir, people_name, config.WC_p_filename)).readlines()]
        }

    def load_landmark_file(landmarks_dir, people_name, image_name):
        '''
        return a list of landmarks base on given parameters.
        '''
        people_name = people_name.replace('_', ' ')
        image_name += '.txt'
        return [tuple(map(int, map(float, landmark.strip().split(' '))))\
                for landmark in\
                open(os.path.join(landmarks_dir, people_name, image_name)).readlines()]

    people_names = [people_name.replace(' ', '_') for people_name in os.listdir(filenames_dir)]

    image_names, landmarks = {}, {}
    for people_name in tqdm(people_names):
        
        image_names[people_name] = get_filenames(filenames_dir, people_name)

        landmarks[people_name] = {}
        for image_type in ('c', 'p',):
            for image_name in image_names[people_name][image_type]:
                landmarks[people_name][image_name] =\
                load_landmark_file(landmarks_dir, people_name, image_name)

    return people_names, image_names, landmarks


@lru_cache(maxsize=config.cache_size)
def get_image(dataset_name, people_name, image_name, show_landmarks=1):
    '''
    return a jpg image buffer based on people_name, image_name and show_landmarks
    '''
    def genarate_landmark_image(src, dst, lamdmark):
        import cv2, numpy as np

        image = cv2.imread(src)
        for ld in lamdmark:
            x, y = ld
            image[y-5:y+5, x-5:x+5, :] = np.array([0, 100, 0])
        cv2.imwrite(dst, image)

    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)
    
    landmark = landmarks[people_name][image_name]
    people_name = people_name.replace('_', ' ')
    image_name += '.jpg'
    image_path = os.path.join(images_dir, people_name, image_name)

    if int(show_landmarks) == 1:
        ld_images_dir = os.path.join(images_dir, config.datasetviewer_dir_name, people_name)
        os.makedirs(ld_images_dir, exist_ok=True)

        ld_images_path = os.path.join(ld_images_dir, image_name)
        if not os.path.exists(ld_images_path):
            genarate_landmark_image(image_path, ld_images_path, landmark)
        image_path = ld_images_path

    return open(image_path, 'rb').read()