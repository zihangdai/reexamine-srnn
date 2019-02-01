import os, sys
import glob
import math
import time

import numpy as np
import torch


class IamondbData(object):
    def __init__(self, data, bsz, device='cpu', Xmean=None, Xstd=None):
        """
            data -- list[Tensor] -- there is no order among the Tensors
        """

        self.bsz = bsz
        data_new = []
        tot_data = []
        for item in data:
            tmp = torch.zeros(item.size(0)-1, 3)
            tmp[:,1:] = item[1:, 1:] - item[:-1, 1:]
            tot_data.append(tmp[:,1:])
            tmp[:,0] = item[1:,0]
            data_new.append(tmp)
        self.data = data_new
        
        self.Xmean, self.Xstd = self.normalize_2(tot_data, Xmean, Xstd)
        print('Xmean: {}, Xstd: {}'.format(self.Xmean, self.Xstd))
        self.device = device
        
    def normalize_1(self, tot_data):
        X = tot_data
        X_len = np.array([len(x) for x in X]).sum()
        X_mean = np.array([x.sum() for x in X]).sum() / X_len
        X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
        X_std = np.sqrt(X_sqr - X_mean**2)    
        return X_mean, X_std
    
    def normalize_2(self, tot_data, Xmean, Xstd):
        tot_data = torch.cat(tot_data, 0)
        if Xmean is None:
            Xmean = tot_data.mean(0, keepdim = True)
        if Xstd  is None:
            Xstd = tot_data.std(0, keepdim = True)
        for i in range(len(self.data)):
            self.data[i][:, 1:] = (self.data[i][:, 1:] - Xmean)/Xstd
        return Xmean, Xstd


    def get_masked_iter(self, shuffle=False, sort=False, distributed=False):
        if shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            if sort:
                indices = np.argsort([len(d) for d in self.data])
            else:
                indices = np.array(range(len(self.data)))
        assert not (shuffle and sort), 'cannot be both `shuffle` and `sort`'
        
        if distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
        else:
            world_size = 1
            local_rank = 0


        for i in range(0, len(indices), self.bsz):
            indices_batch = indices[i+local_rank:i+self.bsz:world_size]
            if len(indices_batch) > 0:
                data_list = [self.data[idx] for idx in indices_batch]
                max_len = max([len(d) for d in data_list])

                # create data [max_len x bsz x dim] and mask [max_len x bsz x 1]
                data_batch = torch.zeros(max_len, len(data_list), 3)
                mask_batch = torch.zeros(max_len, len(data_list))
                for j, d in enumerate(data_list):
                    data_batch[:len(d),j] = d
                    mask_batch[:len(d),j] = 1
                inp = data_batch[:-1].to(self.device)
                tgt = data_batch[1:].to(self.device)
                mask = mask_batch[1:].to(self.device)
                yield inp, tgt, mask


if __name__ == '__main__':
    data_path = '../handwriting/iamondb/'
    tr_data = torch.load(os.path.join(data_path, 'train.t7'))
    tr_data = IamondbData(tr_data, 32, Xmean = 8., Xstd = 56.04462)
    for item in tr_data.data:
        max_x = torch.max(item[:,1])
        max_y = torch.max(item[:,2])
        min_x = torch.min(item[:,1])
        min_y = torch.min(item[:,2])
    va_data = torch.load(os.path.join(data_path, 'valid.t7'))
    va_data = IamondbData(va_data, 32, Xmean=tr_data.Xmean, Xstd=tr_data.Xstd)         