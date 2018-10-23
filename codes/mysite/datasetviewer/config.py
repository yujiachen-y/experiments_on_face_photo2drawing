import os

# cache setting
cache_size = None

# datasets
datasets_dir = os.path.join('..', '..', 'datasets')

# datasetviewer
datasetviewer_dir_name = '.datasetviewer'
backup_scr_dir = os.path.split(__file__)[0]
dataset_config_name = 'config.json'

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


if __name__ == '__main__':
    print(os.listdir(datasets_dir))