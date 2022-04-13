import os
import torch
import torch.distributed as dist
import torch.nn as nn
from utils.bmuf import copy_vec_to_param
from utils.bmuf import SUCCESS, STOP


class BmufTrainer(object):
    def __init__(self, model, dp_group, mp_group, dp_master_node,
                 world_master_node, block_momentum, block_lr):
        self.model = model
        self.dp_group = dp_group
        self.mp_group = mp_group
        self.dp_master_node = dp_master_node
        self.master_node = world_master_node
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dp_size = dist.get_world_size(group=dp_group)
        self.mp_size = dist.get_world_size(group=mp_group)
        assert self.world_size % self.dp_size == 0
        self.expert_world_size = self.world_size // self.dp_size
        self.expert_rank = self.rank % self.expert_world_size
        self.block_momentum = block_momentum
        self.block_lr = block_lr

        # get param list to be synced
        self.sync_params_world = []
        self.sync_params_dp = []
        for p in self.model.parameters():
            if hasattr(p, 'dp_comm') and p.dp_comm == 'mp':
                self.sync_params_dp.append(p)
            else:
                self.sync_params_world.append(p)
        param_vec_world = nn.utils.parameters_to_vector(self.sync_params_world)
        param_vec_dp = nn.utils.parameters_to_vector(self.sync_params_dp)
        self.param_world = param_vec_world.data.clone()
        self.param_dp = param_vec_dp.data.clone()
        # sync params before training
        dist.broadcast(self.param_world, src=self.master_node, async_op=False)
        dist.broadcast(self.param_dp, src=self.dp_master_node, group=self.dp_group, async_op=False)
        # block delta prev
        num_param_world = self.param_world.numel()
        if self.rank == self.master_node:
            self.delta_prev_world = torch.FloatTensor([0] * num_param_world).cuda()
        else:
            self.delta_prev_world = None
            copy_vec_to_param(self.param_world, self.sync_params_world)
        num_param_dp = self.param_dp.numel()
        if self.rank == self.dp_master_node:
            self.delta_prev_dp = torch.FloatTensor([0] * num_param_dp).cuda()
        else:
            self.delta_prev_dp = None
            copy_vec_to_param(self.param_dp, self.sync_params_dp)

    def update_and_sync(self):
        # data parallel group
        delta_dp = self.param_dp - nn.utils.parameters_to_vector(self.sync_params_dp).data
        dist.reduce(tensor=delta_dp, dst=self.dp_master_node, group=self.dp_group)
        if torch.isnan(delta_dp).sum().item():
            return STOP
        if self.rank == self.dp_master_node:
            delta_dp = delta_dp / float(self.dp_size)
            self.delta_prev_dp = self.block_momentum * self.delta_prev_dp + \
                                    self.block_lr * (1 - self.block_momentum) * delta_dp
            self.param_dp -= (1 + self.block_momentum) * self.delta_prev_dp
        dist.broadcast(tensor=self.param_dp, src=self.dp_master_node, group=self.dp_group)
        copy_vec_to_param(self.param_dp, self.sync_params_dp)
        # world parallel
        delta = self.param_world - nn.utils.parameters_to_vector(self.sync_params_world).data
        dist.reduce(tensor=delta, dst=self.master_node)
        if torch.isnan(delta).sum().item():
            return STOP
        if self.rank == self.master_node:
            delta = delta / float(self.world_size)
            self.delta_prev_world = self.block_momentum * self.delta_prev_world + \
                                        self.block_lr * (1 - self.block_momentum) * delta
            self.param_world -= (1 + self.block_momentum) * self.delta_prev_world
        dist.broadcast(tensor=self.param_world, src=self.master_node)
        copy_vec_to_param(self.param_world, self.sync_params_world)
        return SUCCESS

    def state_dict_comm(self):
        state_dict = {'block_momentum': self.block_momentum, 'block_lr': self.block_lr}
        if self.rank == self.master_node:
            state_dict['delta_prev_world'] = self.delta_prev_world
        # model parallel delta prev
        if self.rank == self.dp_master_node:
            pointer = 0
            reduce_deltas = []
            for param in self.sync_params_dp:
                numel = param.numel()
                param_delta = self.delta_prev_dp[pointer:pointer+numel].view_as(param).data
                pointer += numel
                new_size = list(param_delta.size())
                num_exp = new_size[0]
                new_size[0] = num_exp * self.expert_world_size
                reduce_param_delta = param_delta.new_zeros(*new_size)
                reduce_param_delta[self.expert_rank * num_exp: (self.expert_rank + 1) * num_exp] = param_delta
                dist.all_reduce(reduce_param_delta, group=self.mp_group, async_op=False)
                reduce_param_delta = reduce_param_delta.cpu()
                reduce_deltas += [reduce_param_delta]
            delta_vec = nn.utils.parameters_to_vector(reduce_deltas)
            state_dict['delta_prev_dp'] = delta_vec
        return state_dict

    def load_state_dict_comm(self, state_dict):
        self.block_momentum = state_dict['block_momentum']
        self.block_lr = state_dict['block_lr']
        if self.rank == self.dp_master_node:
            delta_prev_dp = state_dict['delta_prev_dp']
            pointer = 0
            delta_list = []
            for param in self.sync_params_dp:
                numel = param.numel() * self.expert_world_size
                new_size = list(param.size())
                num_exp = new_size[0]
                new_size[0] = num_exp * self.expert_world_size
                reduce_param = delta_prev_dp[pointer: pointer+numel].view(new_size)
                pointer += numel
                delta_list += [reduce_param[self.expert_rank*num_exp: (self.expert_rank+1)*num_exp]]
            delta_dp_vec = nn.utils.parameters_to_vector(delta_list)
            self.delta_prev_dp.copy_(delta_dp_vec)
        if self.rank == self.master_node:
            self.delta_prev_world.copy_(state_dict['delta_prev_world'])

    def reset_param_vectors(self):
        self.param_world.copy_(nn.utils.parameters_to_vector(self.sync_params_world).data)
        self.param_dp.copy_(nn.utils.parameters_to_vector(self.sync_params_dp).data)

    def broadcast(self, tensor):
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        dist.reduce(tensor=tensor, dst=self.master_node, async_op=False)


def prepare_moe_info():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # number of workers on which experts are distributed
    expert_world_size = int(os.environ["expert_world_size"])
    assert expert_world_size <= world_size and world_size % expert_world_size == 0
    expert_rank = rank % expert_world_size
    group_idx = rank // expert_world_size
    # model parallel group
    mp_groups = []
    for i in range(0, world_size, expert_world_size):
        mp_ranks = list(range(i, i + expert_world_size))
        mp_groups += [dist.new_group(ranks=mp_ranks)]
    mp_group = mp_groups[group_idx]
    # data parallel group for expert network
    dp_groups = []
    for exp_i in range(expert_world_size):
        dp_ranks = [exp_i + group_offset for group_offset in range(0, world_size, expert_world_size)]
        dp_groups += [dist.new_group(ranks=dp_ranks)]
    dp_group = dp_groups[expert_rank]
    return expert_rank, expert_world_size, mp_group, dp_group
