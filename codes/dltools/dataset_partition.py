import json
import os
import sys

import cv2

from ..mysite.datasetviewer import config
from ..mysite.datasetviewer.datas import get_dirs, get_image, get_overview
from ..mysite.datasetviewer.utils import im_str_to_np, tqdm


def dataset_partition(src, dst, test_people, shape):
    def make_dataset(dataset_dir, chosen_people):
        os.makedirs(dataset_dir, exist_ok=True)
        for dir_name in ('c', 'p',):
            images_dir = os.path.join(dataset_dir, dir_name, 'images')
            landmarks_dir = os.path.join(dataset_dir, dir_name, 'landmarks')
            information_dir = os.path.join(dataset_dir, dir_name, 'informations')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(landmarks_dir, exist_ok=True)
            os.makedirs(information_dir, exist_ok=True)

            for people_name in tqdm(chosen_people):
                info = []
                for image_name in image_names[people_name][dir_name]:
                    new_name = people_name + '_' + image_name
                    info.append(new_name)
                    
                    im = get_image(src, people_name, image_name, show_landmark=False)
                    im = im_str_to_np(im)
                    assert im.shape == (512, 512, 3), 'invalid shape {}'.format(im.shape)
                    im = cv2.resize(im, shape)
                    if not cv2.imwrite(os.path.join(images_dir, new_name+'.jpg'), im):
                        assert 0, 'cv2.imwrite error'
                    
                    with open(os.path.join(landmarks_dir, new_name+'.json'), 'w') as f:
                        json.dump(landmarks[people_name][image_name], f)

                with open(os.path.join(information_dir, people_name+'.json'), 'w') as f:
                    json.dump(info, f)
    
    shape = (shape, shape)
    dst_dir = os.path.join(config.datasets_dir, dst)
    (images_dir, filenames_dir, landmarks_dir), messages = get_dirs(src)
    people_names, image_names, landmarks = get_overview(images_dir, filenames_dir, landmarks_dir)

    people = {'train': [], 'test': [],}
    for people_name in people_names:
        people['test' if people_name in test_people else 'train'].append(people_name)
    for dataset_type in ('train', 'test',):
        make_dataset(os.path.join(dst_dir, dataset_type), people[dataset_type])

    json.dump(test_people, open(os.path.join(dst_dir, 'test_people.json'), 'w'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()
    parse.add_argument('-s', '--src')
    parse.add_argument('-d', '--dst')
    parse.add_argument('-t', '--test_people')
    parse.add_argument('--shape', default=512, type=int)
    args = parse.parse_args()

    test_people = json.load(open(args.test_people))
    os.chdir(os.path.join(os.path.split(__file__)[0], '..', 'mysite'))
    dataset_partition(args.src, args.dst, test_people, args.shape)