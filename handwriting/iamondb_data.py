from __future__ import division
import os
import numpy as np
import fnmatch
import re
from lxml import etree
import torch
import cPickle
import argparse

def fetch_iamondb(data_path):

    strokes_path = os.path.join(data_path, "lineStrokes")
    ascii_path = os.path.join(data_path, "ascii")
    train_files_path = os.path.join(data_path, "train.txt")
    valid_files_path = os.path.join(data_path, "valid.txt")
    
    if not os.path.exists(strokes_path) or not os.path.exists(ascii_path):
        raise ValueError("You must download the data from IAMOnDB, and"
                         "unpack in %s" % data_path)

    if not os.path.exists(train_files_path) or not os.path.exists(valid_files_path):
        raise ValueError("Cannot find concatenated train.txt and valid.txt"
                         "files! See the README in %s" % data_path)

    partial_path = data_path

    def construct_ascii_path(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        file_path = os.path.join(ascii_path, primary_dir, sub_dir, f + ".txt")

        return file_path

    def construct_stroke_paths(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        files_path = os.path.join(strokes_path, primary_dir, sub_dir)

        #Dash is crucial to obtain correct match!
        files = fnmatch.filter(os.listdir(files_path), f + "-*.xml")
        files = [os.path.join(files_path, fi) for fi in files]
        files = sorted(files, key=lambda x: int(x.split(os.sep)[-1].split("-")[-1][:-4]))

        return files

    train_npy_x = os.path.join(partial_path, "train_npy_x.npy")
    train_npy_y = os.path.join(partial_path, "train_npy_y.npy")
    valid_npy_x = os.path.join(partial_path, "valid_npy_x.npy")
    valid_npy_y = os.path.join(partial_path, "valid_npy_y.npy")

    if not os.path.exists(train_npy_x):
        train_names = [f.strip()
                       for f in open(train_files_path, mode='r').readlines()]
        valid_names = [f.strip()
                       for f in open(valid_files_path, mode='r').readlines()]
        train_ascii_files = [construct_ascii_path(f) for f in train_names]
        valid_ascii_files = [construct_ascii_path(f) for f in valid_names]

        train_stroke_files = [construct_stroke_paths(f) for f in train_names]
        valid_stroke_files = [construct_stroke_paths(f) for f in valid_names]
        train_set = (zip(train_stroke_files, train_ascii_files),
                     train_npy_x, train_npy_y)

        valid_set = (zip(valid_stroke_files, valid_ascii_files),
                     valid_npy_x, valid_npy_y)
        for se, x_npy_file, y_npy_file in [train_set, valid_set]:
            x_set = []
            y_set = []
            se = list(se)
            for n, (strokes_files, ascii_file) in enumerate(se):
                if n % 100 == 0:
                    print("Processing file %i of %i" % (n, len(se)))
                with open(ascii_file) as fp:
                    cleaned = [t.strip() for t in fp.readlines()
                               if t != '\r\n'
                               and t != '\n'
                               and t != ' \r\n']

                    # Try using CSR
                    idx = [n for
                           n, li in enumerate(cleaned) if li == "CSR:"][0]
                    cleaned_sub = cleaned[idx + 1:]
                    corrected_sub = []

                    for li in cleaned_sub:
                        # Handle edge case with %%%%% meaning new line?
                        if "%" in li:
                            li2 = re.sub('\%\%+', '%', li).split("%")
                            li2 = [l.strip() for l in li2]
                            corrected_sub.extend(li2)
                        else:
                            corrected_sub.append(li)

                n_one_hot = 57
                y = [np.zeros((len(li), n_one_hot), dtype='int16')
                     for li in corrected_sub]

                # A-Z, a-z, space, apostrophe, comma, period
                charset = list(range(65, 90 + 1)) + list(range(97, 122 + 1)) + [32, 39, 44, 46]
                tmap = {k: n + 1 for n, k in enumerate(charset)}

                # 0 for UNK/other
                tmap[0] = 0

                def tokenize_ind(line):

                    t = [ord(c) if ord(c) in charset else 0 for c in line]
                    r = [tmap[i] for i in t]

                    return r

                for n, li in enumerate(corrected_sub):
                    y[n][np.arange(len(li)), tokenize_ind(li)] = 1

                x = []

                for stroke_file in strokes_files:
                    with open(stroke_file) as fp:
                        tree = etree.parse(fp)
                        root = tree.getroot()
                        # Get all the values from the XML
                        # 0th index is stroke ID, will become up/down
                        s = np.array([[i, int(Point.attrib['x']),
                                      int(Point.attrib['y'])]
                                      for StrokeSet in root
                                      for i, Stroke in enumerate(StrokeSet)
                                      for Point in Stroke])

                        # flip y axis
                        s[:, 2] = -s[:, 2]

                        # Get end of stroke points
                        c = s[1:, 0] != s[:-1, 0]
                        ci = np.where(c == True)[0]
                        nci = np.where(c == False)[0]

                        # set pen down
                        s[0, 0] = 0
                        s[nci, 0] = 0

                        # set pen up
                        s[ci, 0] = 1
                        s[-1, 0] = 1
                        x.append(s)

                if len(x) != len(y):
                    print("Dataset error - len(x) !+= len(y)!")
                    from IPython import embed; embed()
                    raise ValueError()

                x_set.extend(x)
                y_set.extend(y)
            cPickle.dump(x_set, open(x_npy_file, mode="wb"))
            cPickle.dump(y_set, open(y_npy_file, mode="wb"))
    train_x = cPickle.load(open(train_npy_x, mode="rb"))
    train_y = cPickle.load(open(train_npy_y, mode="rb"))
    valid_x = cPickle.load(open(valid_npy_x, mode="rb"))
    valid_y = cPickle.load(open(valid_npy_y, mode="rb"))

    return (train_x, train_y, valid_x, valid_y)

parser = argparse.ArgumentParser(description='Preprocess iamondb for pytorch')
parser.add_argument('--data_dir', required=True, type=str,
                    help='data directory.')
parser.add_argument('--save_dir', required=True, type=str,
                    help='save directory.')

args = parser.parse_args()

data_path = args.data_dir
train_x, _, valid_x, _ = fetch_iamondb(data_path)
for i in range(len(train_x)):
    train_x[i] = torch.from_numpy(train_x[i])
for i in range(len(valid_x)):
    valid_x[i] = torch.from_numpy(valid_x[i])

torch.save(train_x, os.path.join(args.save_dir, 'train.t7'))
torch.save(valid_x, os.path.join(args.save_dir, 'valid.t7'))