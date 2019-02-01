import os, sys
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote
import pickle

import torch

URLs = [
    "http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle",
    "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle",
]

FNs = [
    "nottingham",
    "muse",
]

DIM = 88
MIN = 21

def download_data(args):
    for fn, url in zip(FNs, URLs):

        save_dir = os.path.join(args.save_dir, fn)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pkl = os.path.join(save_dir, "{}.pkl".format(fn))

        if not os.path.exists(pkl):
            os.system("wget {} -O {}".format(url, pkl))
        else:
            print("{} already exists.".format(pkl))

def preprocess_data(args):
    for fn in FNs:
        save_dir = os.path.join(args.save_dir, fn)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pkl = os.path.join(save_dir, "{}.pkl".format(fn))
        dataset = pickle.load(open(pkl, "rb"), encoding="latin1")

        print("=" * 80)
        for split in dataset.keys():
            print("Process {} set of {}".format(split, fn))
            data = []
            for seq in dataset[split]:
                tensor = torch.zeros(len(seq), DIM)
                for idx, frame in enumerate(seq):
                    for val in frame:
                        tensor[idx, val - MIN] = 1
                data.append(tensor)

            total_len = sum([d.size(0) for d in data])
            print(">>> Finish: {} seqs, tot len {}, avg len {:.2f}".format(
                len(data), total_len, total_len / len(data)))

            savepath = os.path.join(save_dir, "{}.{}.t7".format(fn, split))
            torch.save(data, savepath)

def main(args):
    download_data(args)
    preprocess_data(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess midi data')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='location to save the data')
    args = parser.parse_args()

    main(args)
