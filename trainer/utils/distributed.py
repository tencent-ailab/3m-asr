import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPWrapper(DDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, param_dict):
        return self.module.load_state_dict(param_dict)

    def __getattr__(self, name):
        try:
            attr = super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
        return attr


class DistributedGroupedDataParallel(nn.Module):
    def __init__(self, module, dp_group, dp_master_node,
                 world_master_node):
        super().__init__()
        self.module = module
        # data parallel group for expert network
        self.dp_group = dp_group
        self.dp_master_node = dp_master_node
        self.master_node = world_master_node
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dp_size = dist.get_world_size(group=dp_group)
        assert self.world_size % self.dp_size == 0
        self.expert_world_size = self.world_size // self.dp_size
        self.expert_rank = self.rank % self.expert_world_size
        self.comm_types = ['mp', 'dp_mean']
        self._sync_params()

    def _sync_params(self):
        groups = dict()
        for p in self.module.parameters():
            if hasattr(p, "dp_comm"):
                dp_comm = p.dp_comm
            else:
                dp_comm = "dp_mean"
            group_key = (dp_comm, p.dtype)
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, _), group in groups.items():
            if dp_comm not in self.comm_types:
                continue
            if dp_comm == "mp":
                master_node = self.dp_master_node
                comm_group = self.dp_group
            else:
                master_node = self.master_node
                comm_group = None  # default distributed group
            datas = [p.data for p in group]
            coalesced = _flatten_dense_tensors(datas)
            dist.broadcast(coalesced, src=master_node,
                group=comm_group, async_op=False)
            # torch.cuda.synchronize()
            synced = _unflatten_dense_tensors(coalesced, datas)
            for d, s in zip(datas, synced):
                d.copy_(s)

    def allreduce_grad(self):
        groups = dict()
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if hasattr(p, "dp_comm"):
                dp_comm = p.dp_comm
            else:
                dp_comm = "dp_mean"
            group_key = (dp_comm, p.dtype)
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, _), group in groups.items():
            if dp_comm not in self.comm_types:
                continue
            if dp_comm == "mp":
                comm_group = self.dp_group
            else:
                comm_group = None
            grads = [p.grad.data for p in group]
            coalesced = _flatten_dense_tensors(grads)
            if dp_comm == "mp":
                coalesced /= self.dp_size
            else:
                coalesced /= self.world_size
            dist.all_reduce(coalesced, group=comm_group, async_op=False)
            synced = _unflatten_dense_tensors(coalesced, grads)
            for g, s in zip(grads, synced):
                g.copy_(s)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, param_dict):
        return self.module.load_state_dict(param_dict)

    def state_dict_comm(self):
        return self.module.state_dict_comm()

    def load_state_dict_comm(self, param_dict):
        return self.module.load_state_dict_comm(param_dict)

    def __getattr__(self, name):
        try:
            attr = super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
        return attr
