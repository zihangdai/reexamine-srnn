import os, sys
import math
import time
import timeit
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
from distributed_utils import is_master

from models import rnn, rnn_hier_inp
from models import srnn, srnn_hier_inp
from models import rnn_random
from utils.experiment import get_logger, create_exp_dir

import options

def adjust_lr(optimizer, epoch, total_epoch, init_lr, end_lr):
  if (epoch > total_epoch):
    return end_lr

  mult = 0.5 * (1 + math.cos(math.pi * float(epoch) / total_epoch))
  lr = end_lr + (init_lr - end_lr) * mult
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def adjust_kd(epoch, total_epoch, init_kd, end_kd):
  if (epoch > total_epoch):
    return end_kd

  mult = math.cos(0.5 * math.pi * float(epoch) / total_epoch)
  return end_kd + (init_kd - end_kd) * mult

def evaluate(eval_data, model, args):
  model.eval()

  loss_sum = torch.Tensor([0.]).to(args.device)
  cnt = torch.Tensor([0.]).to(args.device)

  with torch.no_grad():
    eval_iter = eval_data.get_masked_iter(distributed=args.distributed)
    for data, mask in eval_iter:
      # refresh the hidden for each test sequence batch
      if args.pass_h:
        eval_hid = model.init_hidden(data.size(1))
      else:
        eval_hid = None

      # for each chunk/segment within the entire sequence
      for x_, y_, x_mask_ in eval_data.yield_chunks(data, mask):
        if args.kld:
          nll_loss, kld_loss, eval_hid = \
            model(x_, y_, mask=x_mask_, hidden=eval_hid)
          nll_loss = nll_loss.sum(0).sum()
          kld_loss = kld_loss.sum(0).sum()
          total_loss = nll_loss - kld_loss
          total_loss = total_loss.detach()
        else:
          nll_loss, eval_hid = \
            model(x_, y_, mask=x_mask_, hidden=eval_hid)
          nll_loss = nll_loss.sum(0).sum()
          total_loss = nll_loss.detach()

        loss_sum += total_loss
        if args.eval_len != -1:
          cnt += x_mask_.sum() * args.d_data

      if args.eval_len == -1:
        cnt += data.size(1)

  model.train()
  if args.distributed:
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(cnt, op=dist.ReduceOp.SUM)

  ret = -loss_sum.item() / cnt.item()
  if args.eval_len != -1:
    ret = ret * args.eval_len

  return ret


def main(args):
  args = options.set_default_args(args)

  if args.ddp_backend == 'apex':
    from apex.parallel import DistributedDataParallel as DDP
  else:
    from torch.nn.parallel import DistributedDataParallel as DDP

  ############################################################################
  # Random seed
  ############################################################################
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  ############################################################################
  # Experiment & Logging
  ############################################################################
  if is_master(args):
    if args.resume:
      # rank-0 device creates experiment dir and log to the file
      logging = get_logger(os.path.join(args.expname, 'log.txt'),
                 log_=not args.debug)
    else:
      # rank-0 device creates experiment dir and log to the file
      logging = create_exp_dir(args.expname, debug=args.debug)
  else:
    # other devices only log to console (print) but not the file
    logging = get_logger(log_path=None, log_=False)

  args.model_path = os.path.join(args.expname, 'model.pt')
  args.var_path = os.path.join(args.expname, 'var.pt')

  ############################################################################
  # Load data
  ############################################################################
  logging('Loading data..')
  tr_data, va_data, te_data = options.load_data(args)

  train_step = 0
  best_eval_ll = -float('inf')
  if args.resume:
    logging('Resuming from {}...'.format(args.resume))
    model, opt = torch.load(args.model_path, map_location='cpu')
    model = model.to(args.device)
    for state in opt.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(args.device)
    best_eval_ll, train_step = torch.load(args.var_path)
  else:
    # create new model
    logging('Building model `{}`...'.format(args.model_name))
    if args.model_name in ['srnn']:
      model = eval(args.model_name).Model(args.d_data, args.d_emb,
          args.d_mlp, args.d_rnn, args.d_lat, n_layer=args.n_layer,
          dropout=args.dropout)
    elif args.model_name in ['srnn_hier_inp']:
      model = eval(args.model_name).Model(args.d_data, args.d_emb,
          args.d_mlp, args.d_rnn, args.d_lat, n_layer=args.n_layer,
          dropout=args.dropout, n_low_layer=args.n_low_layer)
    elif args.model_name in ['rnn']:
      model = eval(args.model_name).Model(args.d_data, args.d_emb,
          args.d_rnn, n_layer=args.n_layer, dropout=args.dropout,
          n_mix=args.n_mix)
    elif args.model_name in ['rnn_hier_inp']:
      model = eval(args.model_name).Model(args.d_data, args.d_emb,
          args.d_rnn, n_layer=args.n_layer, dropout=args.dropout,
          n_mix=args.n_mix, n_low_layer=args.n_low_layer)
    elif args.model_name in  ['rnn_random']:
      model = eval(args.model_name).Model(args.d_data, args.d_emb,
          args.d_rnn, n_layer=args.n_layer, dropout=args.dropout,
          n_mix=args.n_mix,
          d_leak=args.d_leak)
    else:
        raise ValueError('unsupported model type {}'.format(args.model_name))

    model = model.to(args.device)

    # create new optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

  if not args.test_only:
    # criterion params and model params
    crit_params, model_params = [], []
    for n, p in model.named_parameters():
        if 'crit' in n:
            crit_params.append(p)
        else:
            model_params.append(p)

    ############################################################################
    # Distributed Data Parallel
    ############################################################################
    if args.distributed:
      if args.ddp_backend == 'apex':
        torch.cuda.set_device(args.distributed_rank)
        para_model = DDP(model)
      else:
        para_model = DDP(model, device_ids=[args.device_id],
                 output_device=args.device_id)
    else:
      para_model = model

    ############################################################################
    # Log args
    ############################################################################
    args.n_crit_param = sum([p.nelement() for p in crit_params])
    if args.model_name in ['srnn_hier_nade', 'rnn_hier_nade']:
      n_model_param = 0
      for n, p in model.named_parameters():
        if n == 'nade_w_0':
          mask = getattr(model, 'mask_0')
          n_eff, n_tot = mask.sum().int().item(), mask.numel()
          n_model_param += p.size(2) * p.size(3) * n_eff
        if n == 'nade_w_0':
          mask = getattr(model, 'mask_0')
          n_eff, n_tot = mask.sum().int().item(), mask.numel()
          n_model_param += n_eff
        else:
          n_model_param += p.nelement()
      args.n_model_param = n_model_param
    else:
      args.n_model_param = sum([p.nelement() for p in model_params])
    args.n_param = args.n_crit_param + args.n_model_param
    if is_master(args):
      logging('=' * 100)
      for k, v in args.__dict__.items():
        logging('  - {} : {}'.format(k, v))
      logging('=' * 100)

    ############################################################################
    # Training
    ############################################################################
    # linear cosine annealing
    kld_weight = min(1., args.init_kld + train_step * args.kld_incr)

    loss_sum = torch.Tensor([0]).to(args.device)
    kld_sum = torch.Tensor([0]).to(args.device)
    nll_sum = torch.Tensor([0]).to(args.device)
    gnorm_sum = 0
    t = timeit.default_timer()
    for epoch in range(args.num_epochs):
      model.train()
      # make sure all data iterators use the same seed to shuffle data
      if args.distributed:
        np.random.seed(args.seed + epoch)
      tr_iter = tr_data.get_concat_iter(distributed=args.distributed)

      #initalize the hidden state
      if args.pass_h:
        hidden = model.init_hidden(args.batch_size)
      else:
        hidden = None

      for x, y in tr_iter:
        opt.zero_grad()
        if args.kld:
          nll_loss, kld_loss, hidden = para_model(x, y, hidden=hidden)
          nll_loss = nll_loss.mean() * args.ratio
          kld_loss = kld_loss.mean() * args.ratio
          train_loss = nll_loss - kld_loss * kld_weight
          train_loss.backward()

          total_loss = nll_loss.detach() - kld_loss.detach()
          kld_sum += -kld_loss.detach()
          nll_sum += nll_loss.detach()
        else:
          nll_loss, hidden = para_model(x, y, hidden=hidden)
          train_loss = nll_loss.mean() * args.ratio
          train_loss.backward()

          total_loss = train_loss.detach()

        if args.clip > 0:
          gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else:
          gnorm = 0
          for n, p in model.named_parameters():
            param_gnorm = p.grad.data.norm(2)
            gnorm += param_gnorm.item() ** 2
          gnorm = gnorm ** (1. / 2)

        opt.step()

        gnorm_sum += gnorm
        loss_sum += total_loss
        train_step += 1

        # lr & kl annealling
        kld_weight = min(1., kld_weight + args.kld_incr)
        adjust_lr(opt, train_step, args.max_step, args.lr, args.end_lr)

        # log training
        if train_step % args.log_interval == 0:
          if args.distributed:
            dist.reduce(loss_sum, dst=0, op=dist.ReduceOp.SUM)
            loss_sum = loss_sum.div_(args.distributed_world_size)
            dist.reduce(nll_sum, dst=0, op=dist.ReduceOp.SUM)
            nll_sum = nll_sum.div_(args.distributed_world_size)
            dist.reduce(kld_sum, dst=0, op=dist.ReduceOp.SUM)
            kld_sum = kld_sum.div_(args.distributed_world_size)

          if is_master(args):
            cur_loss = loss_sum.item() / args.log_interval
            cur_nll = nll_sum.item() / args.log_interval
            cur_kld = kld_sum.item() / args.log_interval
            elapsed = (timeit.default_timer() - t) / 3600
            logging('| total hrs [{:.2f}] | epoch {} step {} ' \
                    '| lr {:8.6f}, klw {:7.5f} | LL {:>9.4f} ' \
                    '| nll_loss {:>7.4f}, kld_loss {:>8.4f} ' \
                    '| gnorm {:.4f}'.format(
              elapsed, epoch, train_step, opt.param_groups[0]['lr'],
              kld_weight, -cur_loss, cur_nll, cur_kld,
              gnorm_sum / args.log_interval))

          loss_sum = torch.Tensor([0]).to(args.device)
          kld_sum = torch.Tensor([0]).to(args.device)
          nll_sum = torch.Tensor([0]).to(args.device)
          gnorm_sum = 0

        # validation
        if train_step % args.eval_interval == 0:
          if args.d_data == 1 and args.dataset in ['vctk', 'blizzard']:
            # always save checkpoint
            if not args.debug and is_master(args):
              torch.save([model, opt], args.model_path)
              torch.save([best_eval_ll, train_step], args.var_path)
          else:
            eval_ll = evaluate(va_data, model, args)
            if is_master(args):
              logging('-' * 120)
              logging('Eval [{}] at step: {} | valid LL: {:>8.4f}'.format(
                  train_step // args.eval_interval, train_step, eval_ll))
              if eval_ll > best_eval_ll:
                best_eval_ll = eval_ll
                if not args.debug:
                  logging('Save checkpoint. ' \
                          'Best valid LL {:>9.4f}'.format(eval_ll))
                  torch.save([model, opt], args.model_path)
                  torch.save([best_eval_ll, train_step], args.var_path)
              logging('-' * 120)

        # Reach maximum training step
        if train_step == args.max_step:
          break
      if train_step == args.max_step:
        break

    if args.d_data == 1 and args.dataset in ['vctk', 'blizzard']:
      eval_ll = evaluate(va_data, model, args)
      if is_master(args):
        logging('-' * 120)
        logging('Eval [{}] | step: {}, LL: {:>8.4f}'.format(
            train_step // args.eval_interval, train_step, eval_ll))
        logging('-' * 120)

  # evaluate the current model
  if not args.distributed:
    model, _ = torch.load(args.model_path, map_location='cpu')
    model = model.to(args.device)
  test_loss = evaluate(te_data, model, args)
  if is_master(args):
    logging('Test -- LL: {:>8.4f}'.format(test_loss))

if __name__ == '__main__':
  args = options.get_train_args()
  args.distributed_world_size = 1
  main(args)
