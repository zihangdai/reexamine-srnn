import os
import torchaudio
import torch
from subprocess import check_output, getoutput, run
import numpy as np
import argparse
import fnmatch

parser = argparse.ArgumentParser(description='Preprocess vctk for pytorch')
parser.add_argument('--data_dir', required=True, type=str,
                    help='data directory.')
parser.add_argument('--index_dir', type=str, default='./datasets/vctk/',
                    help='the directory of file indexes of the train/valid/text speration.')
parser.add_argument('--save_dir', required=True, type=str,
                    help='save directory.')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
index_dir = args.index_dir
with open(os.path.join(index_dir, 'train_files_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'train_files_list.txt'),'w') as output_file:
        output_file.write(content)

with open(os.path.join(index_dir, 'valid_files_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'valid_files_list.txt'),'w') as output_file:
        output_file.write(content)
        
with open(os.path.join(index_dir, 'test_files_list.txt'), 'r') as content_file:
    content = content_file.read()
    with open(os.path.join(save_dir, 'test_files_list.txt'),'w') as output_file:
        output_file.write(content)

#data_dir = '/usr1/glai1/datasets/VCTK/VCTK-Corpus/wav48/'
data_dir = os.path.join(args.data_dir, 'wav48')
data_16k_dir = os.path.join(args.data_dir, 'VCTK-16k')
if not os.path.exists(data_16k_dir):
    os.makedirs(data_16k_dir)

def get_wav_file_list(data_dir):
    files = []
    files_name = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.wav'):
            files.append(os.path.join(root, filename))
            files_name.append(filename)
    return files, files_name

def convert_to_wav(files, files_name):
    cnt = 0
    for i in range(len(files)):
        target_file = os.path.join(data_16k_dir, files_name[i])
        cmd = ["ffmpeg", "-y", "-i", files[i], "-ar", "16000", target_file]
        print(" ".join(cmd))
        run(cmd)
        
def convert_to_tensor(files_name):
    data_list = []
    data_dir = data_16k_dir
    for f in files_name:
        target_file = os.path.join(data_dir, f)
        data, sample_rate = torchaudio.load(target_file, normalization=True)
        file_name = target_file.split('/')[-1] + '.t7';
        torch.save(data, os.path.join(save_dir, file_name))       
        
files, files_name = get_wav_file_list(data_dir)
convert_to_wav(files, files_name)
convert_to_tensor(files_name)


