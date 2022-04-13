import torch
import torch.distributed as dist

from torch.utils.data import DataLoader


class DistributedEvenLoader(object):
    def __init__(self, local_rank, *args, **kwargs):
        assert dist.is_available() and dist.is_initialized()
        self.local_rank = local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device('cuda:{}'.format(self.local_rank))
        self.reset_flag()
        self.loader = DataLoader(*args, **kwargs)

    def reset_flag(self):
        self.done = 0
        self.all_done = False

    def update(self):
        flag = torch.tensor([self.done]).to(device=self.device)
        dist.all_reduce(tensor=flag)
        num_done = flag.item()
        if num_done == self.world_size:
            self.all_done = True

    def __iter__(self):
        while not self.all_done:
            for _, data in enumerate(self.loader):
                self.update()
                if self.all_done:
                    break
                yield data
            self.done = 1
