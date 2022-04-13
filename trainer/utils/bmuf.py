import torch
import torch.distributed as dist
#import torch.distributed.ReduceOp as ReduceOp
import torch.nn as nn

SUCCESS = 1
STOP = 0


def copy_vec_to_param(vec, parameters):
    r"""Copy vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = param.data.copy_(vec[pointer:pointer + num_param]
                                      .view_as(param).data)
        # Increment the pointer
        pointer += num_param


class BmufTrainer(object):
    def __init__(self, master_node, model, block_momentum, block_lr):
        self.master_node = master_node
        self.world_size = dist.get_world_size()
        self.model = model
        self.block_momentum = block_momentum
        self.block_lr = block_lr
        #clone() make sure self.param
        #NOT tied to model parameters
        #data() enforces no grad
        self.rank = dist.get_rank()
        param_vec = nn.utils.parameters_to_vector(model.parameters())
        self.param = param_vec.data.clone()
        #broadcast initial param to other nodes
        dist.broadcast(tensor=self.param, src=master_node, async_op=False)
        num_param = self.param.numel()
        if self.rank == master_node:
            self.delta_prev = torch.FloatTensor([0]*num_param).cuda()
        else:
            self.delta_prev = None
            #nn.utils.vector_to_parameters(self.param.clone(),
            #                              self.model.parameters())
            copy_vec_to_param(self.param, self.model.parameters())

    def update_and_sync(self):
        delta = self.param - \
                nn.utils.parameters_to_vector(self.model.parameters()).data
        #gather block gradients into delta
        #op=ReduceOp.SUM,
        dist.reduce(tensor=delta, dst=self.master_node)
        #check if model params are still healthy
        if torch.isnan(delta).sum().item():
            return STOP
        if self.rank == self.master_node:
            #local rank is master node
            delta = delta / float(self.world_size)
            self.delta_prev = self.block_momentum * self.delta_prev + \
                              (self.block_lr * (1 - self.block_momentum) * delta)
            self.param -= (1 + self.block_momentum) * self.delta_prev
        dist.broadcast(tensor=self.param, src=self.master_node, async_op=False)
        #nn.utils.vector_to_parameters(self.param.clone(),
        #                              self.model.parameters())
        copy_vec_to_param(self.param, self.model.parameters())

        return SUCCESS

    def state_dict(self):
        state_dict = {'block_momentum': self.block_momentum, 'block_lr': self.block_lr}
        if self.rank == self.master_node:
            state_dict['delta_prev'] = self.delta_prev
        return state_dict

    def load_state_dict(self, state_dict):
        self.block_momentum = state_dict['block_momentum']
        self.block_lr = state_dict['block_lr']
        if self.rank == self.master_node:
            self.delta_prev.copy_(state_dict['delta_prev'])

    def reset_param_vectors(self):
        self.param.copy_(nn.utils.parameters_to_vector(self.model.parameters()).data)

    def broadcast(self, tensor):
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        #op=ReduceOp.SUM,
        dist.reduce(tensor=tensor, dst=self.master_node)
