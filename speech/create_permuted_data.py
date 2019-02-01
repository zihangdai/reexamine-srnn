import os, sys
from shutil import copyfile
import glob
import argparse

import torch

SEED = 123456

def permute_data(data, perm):
    if isinstance(data, list):
        return list([permute_data(d, perm) for d in data])
    else:
        if data.dim() == 2:
            data = data.mean(1)

        n_step = data.size(0) // perm.size(0)
        data = data[:n_step * perm.size(0)].view(n_step, perm.size(0))
        data = data[:, perm].view(-1).contiguous()

        return data

def main(args):
    # create new dir
    parent, child = args.data_dir.rstrip('/').rsplit('/', 1)
    save_dir = os.path.join(parent, '{}-permuted-{}'.format(child, args.d_data))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create and save permutation to new dir
    torch.manual_seed(SEED)
    perm = torch.randperm(args.d_data)
    torch.save(perm, os.path.join(save_dir, 'perm-{}.t7'.format(args.d_data)))
    with open(os.path.join(save_dir, 'perm-{}.txt'.format(args.d_data)), 'w') \
        as f:
        f.write(' '.join([str(i) for i in perm.tolist()]))

    # permutate data and save to new dir
    for fn in os.listdir(args.data_dir):
        print(fn)
        src = os.path.join(args.data_dir, fn)
        dst = os.path.join(save_dir, fn)
        if fn.endswith('.txt'):
            copyfile(src, dst)
        elif fn.endswith('.t7'):
            src_data = torch.load(src)
            dst_data = permute_data(src_data, perm)
            torch.save(dst_data, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--d_data', type=int, default=200)

    args = parser.parse_args()

    main(args)
