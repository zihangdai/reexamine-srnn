import os, sys
import fnmatch
import pickle
import argparse

import torch
import numpy as np
import torchaudio

parser = argparse.ArgumentParser(description='Preprocess timit for pytorch')
parser.add_argument('--data_dir', required=True, type=str,
                    help='data directory.')
parser.add_argument('--save_dir', type=str, default='./',
                    help='save directory.')
parser.add_argument('--valid_frac', type=float, default=0.05,
                    help='fraction of valid data from train.')
args = parser.parse_args()

tr_url = 'https://raw.githubusercontent.com/marcofraccaro/srnn/master/timit_train_files.txt'
te_url = 'https://raw.githubusercontent.com/marcofraccaro/srnn/master/timit_test_files.txt'

def get_files(data_dir, pattern):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_files(split):
    files = []
    path = os.path.join(args.data_dir, 'timit_{}_files.txt'.format(split))
    with open(path, 'r') as f:
        for line in f:
            fields = line.split(split)[-1].split('/')
            fn = os.path.join(args.data_dir, split, *fields).lower()
            files.append(fn)
    return files

def convert_to_tensor(files, args):
    data_list = []
    for f in sorted(files):
        data, sample_rate = torchaudio.load(f, normalization=True)
        data_list.append(data.squeeze())

    return data_list

def record_split(files, split):
    with open('{}.txt'.format(split), 'w') as f:
        for fn in files:
            f.write(fn+'\n')

train_files = get_files(os.path.join(args.data_dir, 'TRAIN'), '*.WAV')
test_files = get_files(os.path.join(args.data_dir, 'TEST'), '*.WAV')

import random
random.seed(1234)
random.shuffle(train_files)

num_train = len(train_files)
num_valid = int(num_train * args.valid_frac)
valid_files = train_files[:num_valid]
train_files = train_files[num_valid:]

record_split(train_files, 'train')
record_split(valid_files, 'valid')
record_split(test_files, 'test')

tr_data = convert_to_tensor(train_files, args)
va_data = convert_to_tensor(valid_files, args)
te_data = convert_to_tensor(test_files, args)

print('Avg. len train {:.2f}'.format(np.mean([len(d) for d in tr_data])))
print('Avg. len valid {:.2f}'.format(np.mean([len(d) for d in va_data])))
print('Avg. len test  {:.2f}'.format(np.mean([len(d) for d in te_data])))

assert len(va_data) + len(tr_data) == 4620
assert len(te_data) == 1680

cat_train = torch.cat(tr_data)
mean, std = torch.mean(cat_train.float()), torch.std(cat_train.float())
print('mean {:.8f}, std {:.8f}, max {:.8f}, min {:.8f}'
      .format(mean, std, torch.max(cat_train), torch.min(cat_train)))

# normalize
tr_data = [(data - mean) / std for data in tr_data]
va_data = [(data - mean) / std for data in va_data]
te_data = [(data - mean) / std for data in te_data]

# make sure the exactly same test data is used
make_multiple = lambda x, dim: x[:int(len(x) // dim) * dim]
te_data = [make_multiple(data, 200) for data in te_data]
print('Total len test {}'.format(sum([len(data) for data in te_data])))

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

torch.save(tr_data, os.path.join(args.save_dir, 'timit.train.t7'))
torch.save(va_data, os.path.join(args.save_dir, 'timit.valid.t7'))
torch.save(te_data, os.path.join(args.save_dir, 'timit.test.t7'))
