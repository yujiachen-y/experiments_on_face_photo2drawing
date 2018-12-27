"""
Modified from https://github.com/NVlabs/MUNIT/blob/master/train.py
"""
import argparse
import os
import shutil
import sys

import tensorboardX
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from trainer import Trainer
from utils import (Timer, get_all_data_loaders, get_config, prepare_sub_folder,
                   write_2images, write_html, write_loss)

cudnn.benchmark = True

def main(opts, yield_mode=False):
    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # Setup data loader
    trainer = Trainer(config)
    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b, combine_loader = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
    print('train a images number is', len(train_loader_a.dataset))
    print('train b images number is', len(train_loader_b.dataset))
    print('test a images number is', len(test_loader_a.dataset))
    print('test b images number is', len(test_loader_b.dataset))

    data_loader = combine_loader if config['sup_w'] > 0 else zip(train_loader_a, train_loader_b)

    # Setup logger and output folders
    from git import Repo
    repo = Repo('.')
    model_name = '%s_%s' % (os.path.splitext(os.path.basename(opts.config))[0], str(repo.head.commit))
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:
        for it, (images_a, labels_a), (images_b, labels_b) in zip(train_loader_a, train_loader_b):
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
            labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach() 

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, labels_a, labels_b, config)
                torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
                # fid
                if yield_mode:
                    paths = trainer.yield_mode_sample(test_loader_a.dataset, test_loader_b.dataset,
                                                      image_directory, iterations)
                    paths = list(paths)
                    paths.append(config['afid_path'])
                    paths.append(config['bfid_path'])
                    other_losses = yield paths
                    write_loss(iterations, None, train_writer, other_losses)

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                # sys.exit('Finish training')
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/init.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    # parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    for iterations, images in main(opts):
        pass
