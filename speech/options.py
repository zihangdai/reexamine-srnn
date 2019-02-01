import os, sys
import time
import argparse

import torch

from utils.data_utils import UnorderedSeqData, UnorderedSeqFileData

def get_train_args():
    parser = argparse.ArgumentParser(description='')
    ##### Path
    parser.add_argument('--data_dir', type=str, default='None',
                        help='location of the data')
    parser.add_argument('--dataset', type=str, default='timit',
                        choices=['timit' , 'blizzard', 'vctk'],
                        help='dataset to use')
    parser.add_argument('--expname', type=str, default='None')
    parser.add_argument('--resume', default='', type=str,
                        help='path to the exp dir from which to resume')
    ##### Data
    parser.add_argument('--d_data', type=int, default=88)
    parser.add_argument('--tgt_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--down_sample', type=int, default=1)
    ##### Model
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_name', type=str, default='rnn')
    # shared
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--d_rnn', type=int, default=750)
    parser.add_argument('--d_emb', type=int, default=512)
    parser.add_argument('--n_mix', type=int, default=20)
    # srnn
    parser.add_argument('--d_mlp', type=int, default=512)
    parser.add_argument('--d_lat', type=int, default=256)
    # hier
    parser.add_argument('--n_low_layer', type=int, default=1)
    parser.add_argument('--d_nade', type=int, default=16)
    # rnn-interleave
    parser.add_argument('--chk_len', type=int, default=2)
    # rnn-random
    parser.add_argument('--d_leak', type=int, default=11)
    # truncated-BPTT
    parser.add_argument('--pass_h', action='store_true')
    parser.add_argument('--skip_length', type=int, default=50)
    ##### Loss
    parser.add_argument('--init_kld', type=float, default=0.2)
    parser.add_argument('--kld_incr', type=float, default=0.00005)
    parser.add_argument('--eval_len', type=float, default=-1)
    ##### Training
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--max_step', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--eval_interval', type=int, default=-1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    ##### Distributed
    parser.add_argument('--device_id', default=0, type=int,
                        help='cuda device id. set to -1 for cpu.')
    parser.add_argument('--distributed_init_method', default=None, type=str,
                        help='Distributed group initialization method.')
    parser.add_argument('--distributed_port', default=11111, type=int,
                        help='port number in the distributed training')
    parser.add_argument('--distributed_world_size', default=None, type=int,
                        help='world size in the distributed setting')
    parser.add_argument('--distributed_rank', default=0, type=int,
                        help='local rank in the distributed setting')
    parser.add_argument('--ddp_backend', default='apex', type=str,
                        choices=['pytorch', 'apex'],
                        help='DDP backend')
    parser.add_argument('--distributed_backend', default='nccl', type=str,
                        help='distributed backend')

    args = parser.parse_args()
    return args

def set_default_args(args):
    # stochastic related
    if 'srnn' in args.model_name:
        args.kld = True
    else:
        args.kld = False

    # use per-frame loss for training
    args.ratio = 1 / args.d_data

    # max train steps
    if args.max_step == -1:
        D = {'timit': 40000, 'blizzard': 160000, 'vctk': 80000}
        args.max_step = D[args.dataset]

    # eval-interval
    if args.eval_interval == -1:
        args.eval_interval = 2000

    # batch size
    if args.batch_size == -1:
        D = {'timit': 32, 'blizzard': 128, 'vctk': 128}
        args.batch_size = D[args.dataset]

    # eval len in terms of #frames
    if args.eval_len == -1:
        D = {'timit':-1, 'blizzard':8000, 'vctk':1}
        args.eval_len = D[args.dataset]

    # expname
    if args.resume:
        args.expname = args.resume
    else:
        if args.expname == 'None':
            args.expname = args.dataset + '_exp'
        suffix = time.strftime('%Y%m%d-%H%M%S') + '-{}-{}-{}-{}'.format(
            args.model_name, args.d_data, args.tgt_len, args.max_step)
        args.expname = os.path.join(args.expname, suffix)

    # whether in distributed setting
    args.distributed = args.distributed_world_size > 1

    # whether to use gpu
    if args.device_id >= 0:
        args.device = 'cuda:{}'.format(args.device_id)
    else:
        args.device = 'cpu'

    return args


def load_data(args):

    if args.dataset == 'timit':
        args.tr_path = os.path.join(args.data_dir, 'timit.train.t7')
        args.va_path = os.path.join(args.data_dir, 'timit.valid.t7')
        args.te_path = os.path.join(args.data_dir, 'timit.test.t7')

        tr_data = UnorderedSeqData(torch.load(args.tr_path), args.batch_size,
            args.d_data, args.tgt_len, ext_len=0,
            device=args.device, down_sample=args.down_sample)
        va_data = UnorderedSeqData(torch.load(args.va_path), args.batch_size,
            args.d_data, args.tgt_len, ext_len=0,
            device=args.device, down_sample=args.down_sample)
        te_data = UnorderedSeqData(torch.load(args.te_path), args.batch_size,
            args.d_data, args.tgt_len, ext_len=0,
            device=args.device, down_sample=args.down_sample)

    elif args.dataset == 'blizzard':
        tr_data = UnorderedSeqFileData(args.data_dir,
            'train_file_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)
        va_data = UnorderedSeqFileData(args.data_dir,
            'valid_file_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)
        te_data = UnorderedSeqFileData(args.data_dir,
            'test_file_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)

    elif args.dataset == 'vctk':
        tr_data = UnorderedSeqFileData(args.data_dir,
            'train_files_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)
        va_data = UnorderedSeqFileData(args.data_dir,
            'valid_files_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)
        te_data = UnorderedSeqFileData(args.data_dir,
            'test_files_list.txt', args.batch_size, args.d_data,
            args.tgt_len, ext_len=0, device=args.device, normalize=True,
            distributed=args.distributed, down_sample=args.down_sample)

    return tr_data, va_data, te_data

