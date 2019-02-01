import os, sys
import time
import argparse

import torch

from utils.iamondb_utils import IamondbData


def get_train_args():
    parser = argparse.ArgumentParser(description='')
    ##### Path
    parser.add_argument('--data_dir', type=str, default='None',
                        help='location of the data')
    parser.add_argument('--dataset', type=str, default='iamondb',
                        help='dataset to use')
    parser.add_argument('--expname', type=str, default='None')
    parser.add_argument('--resume', default='', type=str,
                        help='path to the exp dir from which to resume')
    ##### Data
    parser.add_argument('--d_data', type=int, default=3)
    parser.add_argument('--tgt_len', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--down_sample', type=int, default=1)
    ##### Model
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_name', type=str, default='rnn')
    # shared
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--d_rnn', type=int, default=128)
    parser.add_argument('--d_emb', type=int, default=128)
    parser.add_argument('--n_mix', type=int, default=5)
    # srnn
    parser.add_argument('--d_mlp', type=int, default=128)
    parser.add_argument('--d_lat', type=int, default=128)
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
    parser.add_argument('--eval_interval', type=int, default=300)
    parser.add_argument('--log_interval', type=int, default=30)
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
    if args.model_name in ['srnn', 'srnn_zforce', 'srnn_hier']:
        args.kld = True
    else:
        args.kld = False

    args.n_class = None
    if args.max_step == -1:
        D = {'iamondb': 20000}
        args.max_step = D[args.dataset]
    if args.batch_size == -1:
        D = {'iamondb': 32}
        args.batch_size = D[args.dataset]        
    if args.expname == 'None':
        args.expname = args.dataset + '_exp'
    if args.data_dir == 'None':
        D = {'iamondb': '../../handwriting/iamondb'}
        args.data_dir = D[args.dataset]
    if args.eval_len == -1:
        D = {'iamondb':-1.}
        args.eval_len = D[args.dataset]
        
    # whether in distributed setting
    args.distributed = args.distributed_world_size > 1
    # whether to use gpu
    if args.device_id >= 0:
        args.device = 'cuda:{}'.format(args.device_id)
    else:
        args.device = 'cpu'

    return args

def load_data(args):

    args.tr_path = os.path.join(args.data_dir, 'train.t7')
    args.va_path = os.path.join(args.data_dir, 'valid.t7')

    # in order to compare with previous result, we maually set the std as 56.04462 to match previous implementation
    tr_data = IamondbData(torch.load(args.tr_path), args.batch_size, 
        device=args.device, Xstd = 56.04462)
    va_data = IamondbData(torch.load(args.va_path), args.batch_size,  
        device=args.device, Xmean = tr_data.Xmean, Xstd = tr_data.Xstd)
    
    return tr_data, va_data

