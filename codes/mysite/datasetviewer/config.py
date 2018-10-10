import os

# cache setting
cache_size = None

# datasets
datasets_dir = os.path.join('C:\\Users\\yjc56\\Documents\\_My_Self\\学习\\2018下\\graduation_project\\datasets\\')

# datasetviewer
datasetviewer_dir_name = '.datasetviewer'

# WebCaricature datasets
WC_datasets_dir = os.path.join(datasets_dir, 'WebCaricature')
WC_original_dataset_name = 'original_dataset'

WC_landmarks_dir_name = 'FacialPoints'
WC_filenames_dir_name = 'Filenames'
WC_original_images_dir_name = 'OriginalImages'

WC_c_filename = 'file_c.txt'
WC_p_filename = 'file_p.txt'

# useful classes
class DirNotFindError(RuntimeError):
    pass