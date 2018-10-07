import os

from django.shortcuts import render

try:
    from . import config
    from .utils import get_dir, get_overview_from_memory
except ImportError:
    import config
    from utils import get_dir

# Create your views here.
def index(request):
    datasets = os.listdir(config.WC_datasets_dir)
    return render(request, 'datasetviewer/index.html', {'datasets': datasets})


def overview(request, dataset_name):
    original_images_dir = os.path.join(config.WC_datasets_dir,\
        dataset_name, config.WC_original_images_dir_name)
    if not os.path.exists(original_images_dir):
        return render(request, 'datasetviewer/error.html', {
            "error_message": "path: %s doesn't exist" % original_images_dir,
        })
    
    filenames_dir, fd_message = get_dir(dataset_name, config.WC_filenames_dir_name)
    if filenames_dir is None:
        return render(request, 'datasetviewer/error.html', {
            'error_message': fd_message,
        })

    landmarks_dir, ld_message = get_dir(dataset_name, config.WC_landmarks_dir_name)
    if landmarks_dir is None:
        return render(request, 'datasetviewer/error.html', {
            'error_message': ld_message,
        })

    people_names, image_names = get_overview_from_memory(dataset_name, filenames_dir, landmarks_dir)

    return render(request, 'datasetviewer/overview.html', {
        'messages': [fd_message, ld_message],
        'people_names': people_names,
    })


if __name__ == '__main__':
    overview(None, config.WC_original_dataset_name)