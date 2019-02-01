import os, sys
import glob
import math
import time

import numpy as np
import torch


class UnorderedSeqFileData(object):
    def __init__(self, data_dir, file_list, bsz, dim, tgt_len, ext_len=None,
                 device='cpu', normalize=True, distributed=False,
                 down_sample=1):
        self.data_dir = data_dir
        self.file_list = []
        with open(os.path.join(data_dir, file_list), 'r') as fp:
            for line in fp:
                fn = line.strip()
                if not fn.endswith('.t7'):
                    fn = fn + '.t7'
                self.file_list.append(os.path.join(data_dir, fn))

        self.bsz = bsz
        self.dim = dim
        self.down_sample = down_sample
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.normalize = normalize

        if normalize:
            fn = os.path.join(data_dir, 'mean_and_std.txt')
            if distributed:
                local_rank = torch.distributed.get_rank()
            else:
                local_rank = 0

            if local_rank == 0:
                if os.path.isfile(fn):
                    print('Load mean and std from {}'.format(fn))
                    fin = open(fn)
                    inp = fin.read().split()
                    self.mean, self.std = float(inp[0]), float(inp[1])
                else:
                    print('Cache mean and std to {}'.format(fn))
                    self.mean, self.std = self._calc_variance(fn)
            else:
                while not os.path.isfile(fn):
                    time.sleep(5)

                print('Load mean and std from {}'.format(fn))
                fin = open(fn)
                inp = fin.read().split()
                self.mean, self.std = float(inp[0]), float(inp[1])


    def _calc_variance(self, fout):
        mean = 0
        cnt = 0
        for fn in self.file_list:
            data = torch.load(fn)
            mean += data.sum().item()
            cnt += data.numel()
        mean = mean / cnt
        std = 0
        for fn in self.file_list:
            data = torch.load(fn)
            d = data - mean
            std += (d * d).sum().item()
        std = math.sqrt(std / cnt)

        with open(fout, 'w') as fp:
            fp.write('{}\t{}'.format(mean, std))
            fp.flush()

        return mean, std


    def yield_chunks(self, data_batch, mask_batch=None, len_std=0):
        # create inp, tgt & mask (if not None) chunks
        for k in range(0, data_batch.size(0) - 1, self.tgt_len):
            if len_std > 0:
                tgt_len = max(5, int(np.random.normal(self.tgt_len, len_std)))
            else:
                tgt_len = self.tgt_len
            seq_len = min(tgt_len, data_batch.size(0) - 1 - k)
            end_idx = k + seq_len
            beg_idx = max(0, k - self.ext_len)
            inp = data_batch[beg_idx:end_idx].to(self.device)
            tgt = data_batch[k+1:end_idx+1].to(self.device)

            if self.normalize:
                inp = (inp - self.mean) / self.std
                tgt = (tgt - self.mean) / self.std

            if mask_batch is not None:
                mask = mask_batch[k+1:end_idx+1].to(self.device)
                yield inp, tgt, mask
            else:
                yield inp, tgt


    def _load_files(self, files):
        data = []
        for f in files:
            try:
                d = torch.load(f)
            except:
                print('Error loading file {}'.format(f))
                continue
            if self.down_sample > 1:
                tmp = self._reshape(d, self.down_sample)
                tmp = tmp.transpose(0, 1).contiguous()
                tmp = tmp.view(-1, 1)
                d = tmp
            data.append(d)

        return data


    def _reshape(self, x, dim):
        x = x.mean(1)
        seq_len = int(len(x) // dim)
        return x.narrow(0, 0, seq_len * dim).view(seq_len, dim)


    def get_masked_iter(self, shuffle=False, sort=True, distributed=False):
        if distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
        else:
            world_size = 1
            local_rank = 0

        for i in range(0, len(self.file_list), self.bsz):
            local_file_batch = self.file_list[i+local_rank:i+self.bsz:world_size]
            # In distributed seting, higher-rank nodes may not get a sample
            if len(local_file_batch) > 0:
                data_list = [self._reshape(data, self.dim)
                             for data in self._load_files(local_file_batch)]
                max_len = max([len(d) for d in data_list])

                # create data [max_len x bsz x dim] and mask [max_len x bsz x 1]
                data_batch = torch.zeros(max_len, len(data_list), self.dim)
                mask_batch = torch.zeros(max_len, len(data_list))
                for j, d in enumerate(data_list):
                    data_batch[:len(d),j] = d
                    mask_batch[:len(d),j] = 1

                yield data_batch, mask_batch


    def get_concat_iter(self, shuffle=True, len_std=0, buffer_size=None,
                        distributed=False):
        if shuffle:
            indices = np.random.permutation(len(self.file_list))
            file_list = [self.file_list[idx] for idx in indices]
        else:
            file_list = self.file_list

        if distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
        else:
            world_size = 1
            local_rank = 0

        if buffer_size is None:
            buffer_size = self.bsz

        # #### Strategy (1): split files to each distributed worker
        # """
        #     - Each worker get an even split of mutually exclusive files
        #     - Each worker loads `buffer_size` files into a concated tensor
        #     - The concated tensor is reshaped into [n_step x local_bsz x dim]
        #     - Chunks are generated from the reshaped tensor
        # """
        # buffer_size = max(1, buffer_size // world_size)

        # even_size = (len(file_list) // world_size) * world_size
        # local_files = file_list[local_rank:even_size:world_size]
        # local_bsz = self.bsz // world_size

        # for i in range(0, len(local_files), buffer_size):
        #     data = self._load_files(local_files[i:i+buffer_size])
        #     data = torch.cat(data, dim=0)

        #     n_step = data.size(0) // (self.dim * local_bsz)
        #     data = data.narrow(0, 0, local_bsz * n_step * self.dim)
        #     data = data.view(local_bsz, n_step, self.dim)

        #     data = data.transpose(0, 1).contiguous()

        #     for inp, tgt in self.yield_chunks(data, len_std=len_std):
        #         yield inp, tgt

        # #### Strategy (2): concat files and split the concated tensor
        # share_files = file_list[even_size:]
        # if share_files:
        #     data = self._load_files(share_files)
        #     data = torch.cat(data, dim=0)

        #     n_step = data.size(0) // (self.dim * self.bsz)
        #     data = data.narrow(0, 0, self.bsz * n_step * self.dim)
        #     data = data.view(self.bsz, n_step, self.dim)

        #     if distributed:
        #         data = data[local_rank::world_size].transpose(0, 1).contiguous()
        #     else:
        #         data = data.transpose(0, 1).contiguous()

        #     for inp, tgt in self.yield_chunks(data, len_std=len_std):
        #         yield inp, tgt

        #### Simple Strategy: concat files and split the concated tensor
        for i in range(0, len(file_list), buffer_size):
            data = self._load_files(file_list[i:i+buffer_size])
            data = torch.cat(data, dim=0)

            n_step = data.size(0) // (self.dim * self.bsz)
            data = data.narrow(0, 0, self.bsz * n_step * self.dim)
            data = data.view(self.bsz, n_step, self.dim)

            if distributed:
                data = data[local_rank::world_size].transpose(0, 1).contiguous()
            else:
                data = data.transpose(0, 1).contiguous()

            for inp, tgt in self.yield_chunks(data, len_std=len_std):
                yield inp, tgt


class UnorderedSeqData(object):
    def __init__(self, data, bsz, dim, tgt_len, ext_len=None, device='cpu',
                 down_sample=1):
        """
            data -- list[Tensor] -- there is no order among the Tensors
        """
        self.data = data

        self.bsz = bsz
        self.dim = dim

        if down_sample != 1:
            for i in range(len(self.data)):
                tmp = self._reshape(self.data[i], down_sample)
                tmp = tmp.transpose(0, 1).contiguous()
                tmp = tmp.view(-1)
                self.data[i] = tmp

        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

    def _reshape(self, x, dim):
        seq_len = int(len(x) // dim)
        return x.narrow(0, 0, seq_len * dim).view(seq_len, dim)

    def yield_chunks(self, data_batch, mask_batch=None, len_std=0):
        # create inp, tgt & mask (if not None) chunks
        for k in range(0, data_batch.size(0) - 1, self.tgt_len):
            if len_std > 0:
                tgt_len = max(5, int(np.random.normal(self.tgt_len, len_std)))
            else:
                tgt_len = self.tgt_len
            seq_len = min(tgt_len, data_batch.size(0) - 1 - k)
            end_idx = k + seq_len
            beg_idx = max(0, k - self.ext_len)

            inp = data_batch[beg_idx:end_idx].to(self.device)
            tgt = data_batch[k+1:end_idx+1].to(self.device)

            if mask_batch is not None:
                mask = mask_batch[k+1:end_idx+1].to(self.device)
                yield inp, tgt, mask
            else:
                yield inp, tgt

    def get_masked_iter(self, shuffle=False, sort=True, distributed=False):

        assert not (shuffle and sort), 'cannot be both `shuffle` and `sort`'
        if shuffle:
            indices = np.random.permutation(len(self.data))
        else:
            if sort:
                indices = np.argsort([len(d) for d in self.data])
            else:
                indices = np.array(range(len(self.data)))

        if distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
        else:
            world_size = 1
            local_rank = 0

        for i in range(0, len(indices), self.bsz):
            indices_batch = indices[i+local_rank:i+self.bsz:world_size]
            # In distributed seting, higher-rank nodes may not get a sample
            if len(indices_batch) > 0:
                data_list = [self._reshape(self.data[idx], self.dim)
                             for idx in indices_batch]
                max_len = max([len(d) for d in data_list])

                # create data [max_len x bsz x dim] and mask [max_len x bsz x 1]
                data_batch = torch.zeros(max_len, len(data_list), self.dim)
                mask_batch = torch.zeros(max_len, len(data_list))
                for j, d in enumerate(data_list):
                    data_batch[:len(d),j] = d
                    mask_batch[:len(d),j] = 1

                yield data_batch, mask_batch

    def get_concat_iter(self, shuffle=True, len_std=0, distributed=False):
        if shuffle:
            indices = np.random.permutation(len(self.data))
            data = torch.cat([self.data[idx] for idx in indices])
        else:
            data = torch.cat(self.data)

        n_step = data.size(0) // (self.dim * self.bsz)
        data = data.narrow(0, 0, self.bsz * n_step * self.dim)
        data = data.view(self.bsz, n_step, self.dim)

        if distributed:
            """
                Process with rank `i` takes care of the batch subset
                `data_i = data[local_rank::world_size]`

                As a result, the effective batch size for each process is
                `local_bsz = bsz // world_size`
            """
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()

            local_bsz = self.bsz // world_size
            data = data[local_rank::world_size].transpose(0, 1).contiguous()
        else:
            data = data.transpose(0, 1).contiguous()

        data = data.to(self.device)

        return self.yield_chunks(data, len_std=len_std)


class MultiDimSeqData(UnorderedSeqData):
    def __init__(self, data, bsz, dim, tgt_len, ext_len=None, device='cpu'):
        assert dim in [88, 1]
        new_data = []
        for seq in data:
            new_seq = seq.view(-1)
            new_data.append(new_seq)

        super(MultiDimSeqData, self).__init__(new_data, bsz, dim, tgt_len,
                ext_len=ext_len, device=device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unit Test')
    parser.add_argument('--datadir', type=str, default='./',
                        help='location of the data corpus')
    args = parser.parse_args()

    tr_data = torch.load(os.path.join(args.datadir, 'piano.train.t7'))
    va_data = torch.load(os.path.join(args.datadir, 'piano.valid.t7'))

    tr_data = MultiDimSeqData(tr_data, 1, 100)
    va_data = MultiDimSeqData(va_data, 1, 100)

    batch = 0
    for inp, tgt in tr_data.get_concat_iter():
        batch += 1
        print(batch, inp.dtype, inp.size(), tgt.size())

    batch = 0
    for data_batch, mask_batch in va_data.get_masked_iter(sort=True):
        batch += 1
        batch_len = 0
        for inp, tgt, mask in va_data.yield_chunks(data_batch, mask_batch):
            # print(batch, inp.size(), tgt.size(), mask.eq(0).int().sum(), mask.numel())
            batch_len += inp.size(0)

        print(batch_len)

    # tr_data = torch.load('data/timit.train.t7')
    # va_data = torch.load('data/timit.valid.t7')

    # tr_data = UnorderedSeqData(tr_data, 64, 200, 40)
    # va_data = UnorderedSeqData(va_data, 32, 200, 40)

    # batch = 0
    # for inp, tgt in tr_data.get_concat_iter():
    #     batch += 1
    #     print(batch, inp.size(), tgt.size())

    # batch = 0
    # for data_batch, mask_batch in va_data.get_masked_iter(sort=True):
    #     batch += 1
    #     batch_len = 0
    #     for inp, tgt, mask in va_data.yield_chunks(data_batch, mask_batch):
    #         print(batch, inp.size(), tgt.size(), mask.eq(0).int().sum(), mask.numel())
    #         batch_len += inp.size(0)

    #     print(batch_len)

    # dataset = 'blizzard'
    # bsz = 64
    # tgt_len = 4000

    # tr_data = UnorderedSeqFileData('/usr1/glai1/datasets/{}/torch_data'.format(dataset),
    #     'train_file_list.txt', bsz, 1, tgt_len, normalize=False)

    # batch = 0
    # tr_iter = tr_data.get_concat_iter()
    # for inp, tgt in tr_iter:
    #     batch += 1
    #     if inp.size(0) != tgt_len:
    #         print(batch, inp.size(), tgt.size())

    # va_data = UnorderedSeqFileData('/usr1/glai1/datasets/{}/torch_data'.format(dataset),
    #     'valid_file_list.txt', bsz, 1, tgt_len, normalize=False)

    # batch = 0
    # for data_batch, mask_batch in va_data.get_masked_iter(sort=True):
    #     batch += 1
    #     batch_len = 0
    #     for idx, (inp, tgt, mask) in enumerate(va_data.yield_chunks(data_batch, mask_batch)):
    #         if inp.size(1) != bsz or inp.size(0) != tgt_len:
    #             print(batch, idx, inp.size(), tgt.size(), mask.eq(0).int().sum(), mask.numel())
    #         batch_len += inp.size(0)

    #     print(batch_len)
