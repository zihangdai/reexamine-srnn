import os
import torchaudio
import torch
from subprocess import check_output, getoutput, run
import numpy as np
import argparse
import fnmatch

parser = argparse.ArgumentParser(description='Preprocess blizzard for pytorch')
parser.add_argument('--data_dir', required=True, type=str,
                    help='data directory.')
parser.add_argument('--index_dir', type=str, default='./datasets/blizzard/',
                    help='the directory of file indexes of the train/valid/text speration.')
parser.add_argument('--save_dir', required=True, type=str,
                    help='save directory.')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
index_dir = args.index_dir
with open(os.path.join(index_dir, 'train_file_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'train_file_list.txt'),'w') as output_file:
        output_file.write(content)

with open(os.path.join(index_dir, 'valid_file_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'valid_file_list.txt'),'w') as output_file:
        output_file.write(content)
        
with open(os.path.join(index_dir, 'test_file_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'test_file_list.txt'),'w') as output_file:
        output_file.write(content)
        
def get_mp3_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.mp3'):
            files.append(os.path.join(root, filename))
    return files
def convert_to_wav(files):
    for f in sorted(files):
        target_file = f[:-3] + 'wav'
        cmd = ["ffmpeg", "-y", "-i", f, "-ar", "16000", target_file]
        run(cmd)
def convert_to_tensor(files):
    data_list = []
    file_name_set = set()
    length_map = [(0, '')]
    for i, f in enumerate(files):
        target_file = f[:-3] + 'wav'
        data, sample_rate = torchaudio.load(target_file, normalization=True)
        file_name = target_file.split('/')[-1] + '.t7'
        assert file_name not in file_name_set
        file_name_set.add(file_name)
        torch.save(data, os.path.join(save_dir, file_name))
        length_map.append((length_map[-1][0] + data.size(0), file_name))
    return length_map        
        
files = get_mp3_file_list(data_dir)
convert_to_wav(files)
result = convert_to_tensor(files)


