import os
import sys
from argparse import ArgumentParser

import numpy as np

submodule_path = os.path.abspath('./codes/3rdparty/')
try:
    import fid_score
except ImportError:
    sys.path.append(os.path.join(submodule_path, 'pytorch-fid'))
    import fid_score

batch_size = 64
dims = 2048

def save_statistics(src, dst, cuda):
    block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = fid_score.InceptionV3([block_idx])
    if cuda:
        model = model.cuda()

    m, s = fid_score._compute_statistics_of_path(src, model, batch_size, dims, cuda)
    np.savez(dst, mu=m, sigma=s)


def train_with_fid(args, train_main_func):
    train_main = train_main_func(args, yield_mode=True)
    fids = None

    while True:
        try:
            paths = train_main.send(fids)
            fids = {
                'a_recon_fid': fid_score.calculate_fid_given_paths([paths[4], paths[0]],
                                                                   batch_size, args.gpu != '', dims),
                'ab_fid': fid_score.calculate_fid_given_paths([paths[5], paths[1]],
                                                              batch_size, args.gpu != '', dims),
                'b_recon_fid': fid_score.calculate_fid_given_paths([paths[5], paths[2]],
                                                                   batch_size, args.gpu != '', dims),
                'ba_fid': fid_score.calculate_fid_given_paths([paths[4], paths[3]],
                                                              batch_size, args.gpu != '', dims)
            }
        except StopIteration:
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    # gpu
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    # precalculate fid statstics
    parser.add_argument('--precalu', action='store_true',
                        help='precalculate fid statstics')
    parser.add_argument('--fid_src',
                        help='image directory when precalculate fid statstics')
    parser.add_argument('--fid_dst',
                        help='file path which to store mu and sigma when precalculate fid statstics')
    # SATNet
    parser.add_argument('--config', type=str, default='configs/init.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    # parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")s

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.precalu:
        save_statistics(args.fid_src, args.fid_dst, args.gpu != '')
    else :
        from ..SATNet.train import main as train_main_func
        train_with_fid(args, train_main_func)