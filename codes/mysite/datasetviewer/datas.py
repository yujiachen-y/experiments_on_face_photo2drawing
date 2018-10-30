import json
import os
import pickle
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np

from . import config
from .utils import tqdm


@lru_cache(maxsize=config.cache_size)
def get_dataset_config(dataset_name):
    try:
        with open(os.path.join(config.WC_datasets_dir, dataset_name, config.dataset_config_name)) as f:
            return json.load(f)
    except:
        return defaultdict(lambda: config.WC_original_dataset_name)


def get_dirs(dataset_name):
    '''
    give a dataset_name, and return
    (images_dir, filenames_dir, landmarks_dir),
    (fd_message, ld_message)
    or raise an Exception
    '''
    def get_dir(dataset_name, dir_name, alt_dataset_name):
        result_dir = os.path.join(config.WC_datasets_dir,\
            dataset_name, dir_name)
        result_message = None
        if not os.path.exists(result_dir):
            tmp_result_dir = os.path.join(config.WC_datasets_dir,\
                alt_dataset_name, dir_name)
            if not os.path.exists(tmp_result_dir):
                result_message = "path: %s and path: %s doesn't exist" % (result_dir, tmp_result_dir)
                result_dir = None
            else:
                result_message = "path: %s doesn't exits, so we use path: %s instead." % (result_dir, tmp_result_dir)
                result_dir = tmp_result_dir
        return result_dir, result_message

    dataset_config = get_dataset_config(dataset_name)

    images_dir_name = config.WC_original_images_dir_name
    images_dir, id_message = get_dir(dataset_name, images_dir_name, dataset_config[images_dir_name])
    filenames_dir_name = config.WC_filenames_dir_name
    filenames_dir, fd_message = get_dir(dataset_name, filenames_dir_name, dataset_config[filenames_dir_name])
    landmarks_dir_name = config.WC_landmarks_dir_name
    landmarks_dir, ld_message = get_dir(dataset_name, landmarks_dir_name, dataset_config[landmarks_dir_name])

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
            image_names: {people_name: {'c' and 'p': [filename]}},
            landmarks: {people_name: {filename: landmark}}
    '''
    def get_filenames(filenames_dir, people_name):
        '''
        get a people's filenames from a filenames_dir
        return a dict which keys are ('c', 'p',) and values are list of image number like 'C00001'
        '''
        people_name = people_name.replace('_', ' ')
        result = {}
        for image_type in ('c', 'p',):
            if image_type == 'c':
                filename_path = config.WC_c_filename
            elif image_type == 'p':
                filename_path = config.WC_p_filename
            filename_path = os.path.join(filenames_dir, people_name, filename_path)
            if os.path.exists(filename_path):
                with open(filename_path) as f:
                    filenames = [os.path.splitext(filename.strip())[0] for filename in f.readlines() if not filename.strip() == '']
            else:
                filenames = []
            # filenames = [filename for filename in filenames if not filename == '']
            result[image_type] = filenames
        return result

    def load_landmark_file(landmarks_dir, people_name, image_name):
        '''
        return a list of landmarks base on given parameters.
        '''
        people_name = people_name.replace('_', ' ')
        image_name += '.txt'
        return [tuple(map(float, landmark.strip().split(' ')))\
                for landmark in\
                open(os.path.join(landmarks_dir, people_name, image_name)).readlines()]

    people_names = [people_name.replace(' ', '_') for people_name in sorted(os.listdir(filenames_dir))]

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
def get_image(dataset_name, people_name, image_name, show_landmark=1):
    '''
    return a jpg image buffer based on people_name, image_name and show_landmark
    '''
    def genarate_landmark_image(src, dst, lamdmark):
        src = os.path.split(src)
        os.chdir(src[0])
        image = cv2.imread(src[1])
        w, h, _ = image.shape
        for ld in lamdmark:
            x, y = map(round, ld)
            if 5 <= y < w-5 and 5 <= x < h-5:
                image[y-5:y+5, x-5:x+5, :] = np.array([0, 100, 0])
        dst = os.path.split(dst)
        os.chdir(dst[0])
        cv2.imwrite(dst[1], image)

    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)
    
    landmark = landmarks[people_name][image_name]
    people_name = people_name.replace('_', ' ')
    image_name += '.jpg'
    image_path = os.path.join(images_dir, people_name, image_name)

    if int(show_landmark) == 1:
        ld_images_dir = os.path.join(images_dir, config.datasetviewer_dir_name, people_name)
        os.makedirs(ld_images_dir, exist_ok=True)

        ld_images_path = os.path.join(ld_images_dir, image_name)
        if not os.path.exists(ld_images_path):
            genarate_landmark_image(image_path, ld_images_path, landmark)
        image_path = ld_images_path

    return open(image_path, 'rb').read()


@lru_cache(maxsize=config.cache_size)
def get_pose(dataset_name, people_name, image_name):
    '''
    return (pitch, yaw, roll) to represent face's pose
    '''
    from .pose_estimation import pose_estimation

    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)

    image = get_image(dataset_name, people_name, image_name, show_landmark=0)
    landmark = landmarks[people_name][image_name]
    return pose_estimation(image, landmark, type=0)


@lru_cache(maxsize=config.cache_size)
def get_missing_file(dataset_name):
    def get_current_dict(path):
        if os.path.normpath(path) == protocols_path:
            return result
        paths = os.path.split(path)
        t = get_current_dict(paths[0])
        if paths[1] not in t:
            t[paths[1]] = OrderedDict()
        return t[paths[1]]

    def count_missing_file(file_path):
        def get_line_type(words):
            '''
            type 0: 'Luciano Pavarotti	23	27',
            type 1: 'Ringo Starr	P00021',
            type 1 special case: 'Angela Lansbury	C00001	Moe Howard	P00005',
            type 2: 'Angela Lansbury	C00001	P00004'
            '''
            if words[-1][0] not in ('C', 'P',):
                yield 0, words
            elif not str.isdigit(words[-2][1]):
                for i, word in enumerate(words):
                    if str.isdigit(word[1]):
                        yield 1, words[:i+1]
                        if i != len(words) - 1:
                            yield 1, words[i+1:]
                        break
            else:
                yield 2, words

        def is_valid_photo(people_name, photo):
            return people_name in landmarks and photo in landmarks[people_name]

        result, total = 0, 0
        with open(file_path) as f:
            for line in f.readlines():
                words = line.split()
                if len(words) == 1:
                    continue
                for line_type, words in get_line_type(words):
                    last_index = -1 if line_type == 1 else -2
                    people_name = '_'.join(words[:last_index])
                    if line_type == 0:
                        result = result + int(words[-1]) + int(words[-2]) -\
                                 (len(landmarks[people_name]) if people_name in landmarks else 0)
                        total = total + int(words[-1]) + int(words[-2])
                    elif line_type == 1:
                        result = result + 1 - int(is_valid_photo(people_name, words[-1]))
                        total = total + 1
                        # if result != 0:
                        #     from IPython import embed; embed()
                    else:
                        result = result + 2 - int(is_valid_photo(people_name, words[-2])) - int(is_valid_photo(people_name, words[-1]))
                        total = total + 2
        return result, total, result / total

    def statisitic_result(result):
        count, total = result.pop('count', (0, 0))
        for key, value in result.items():
            if type(value) is tuple:
                count += value[0]
                total += value[1]
            else:
                result[key] = statisitic_result(value)
                count += result[key]['count'][0]
                total += result[key]['count'][1]
        result['count'] = (count, total, count / total)
        return result
    
    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(dataset_name)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)

    protocols_path = os.path.normpath(os.path.join(
        config.WC_datasets_dir,
        config.WC_original_dataset_name,
        config.WC_evaluation_protocols_dir_name,
    ))
    file_path = os.path.join(
        protocols_path,
        config.datasetviewer_dir_name,
        dataset_name,
    )
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    from collections import OrderedDict
    result = OrderedDict()
    # import ipdb; ipdb.set_trace()
    for roots, dirs, files in sorted(os.walk(protocols_path)):
        if config.datasetviewer_dir_name in roots:
            continue
        dirs = sorted(dirs)
        t = get_current_dict(roots)
        for f in sorted(files):
            if os.path.splitext(f)[-1] != '.txt':
                continue
            t[f] = count_missing_file(os.path.join(roots, f))

    result = statisitic_result(result)
    file_dir = os.path.split(file_path)[0]
    os.makedirs(file_dir, exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(result, file)
    return result


if __name__ == '__main__':
    print(get_missing_file('corrected_landmark_dataset_v000'))