import os

try:
    from . import config
except ImportError:
    import config


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
    return result_dir, result_message


def get_filenames(filenames_dir, people_name):
    '''
    get a people's filenames from a filenames_dir
    return a dict which keys are ('c', 'p',) and values are list of image number like 'C00001'
    '''
    pass


def load_landmark_file(landmarks_dir, people_name, image_name):
    '''
    return a list of landmarks base on given parameters.
    '''
    pass


def get_overview_from_memory(dataset_name, filenames_dir, landmarks_dir):
    key = (dataset_name, filenames_dir, landmarks_dir)
    if not key in config._memory:
        people_names = [people_name.replace(' ', '_') for people_name in os.listdir(filenames_dir)]
        images = {}
        for people_name in people_names:
            image_names[people_name] = get_filenames(filenames_dir, people_name)
            for image_type in ('c', 'p',):
                for index, image_name in enumerate(image_names[people_name][image_type]):
                    image_names[people_name][image_type][index] =\
                        (image_name, load_landmark_file(landmarks_dir, people_name, image_name))
        config._memory[key] = (people_names, images)
    return config._memory[key]