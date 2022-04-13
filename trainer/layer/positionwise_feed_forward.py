#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Positionwise feed forward layer definition."""

import torch
from fmoe.layers import FMoELinear
from fmoe.functions import moe_prepare_forward
from fmoe.functions import MOEScatter, MOEGather, MOELinear, MOEbiasLinear
from loss.balance_loss import SparseL1Loss, BalanceImportanceLoss


def mark_module_parallel_comm(m, dp_comm):
    for p in m.parameters():
        setattr(p, "dp_comm", dp_comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size,
                                 capacity=-1, comm=None):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = moe_prepare_forward(gate, num_expert, world_size, comm=comm)
    x = MOEScatter.apply(
        inp, pos,
        local_expert_count, global_expert_count, fwd_batch_size, world_size
    )
    x = expert_fn(x, fwd_expert_count, capacity=capacity)
    x = MOEGather.apply(
        x, pos, local_expert_count, global_expert_count, inp.shape[0], world_size
    )
    return x


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class Expert(torch.nn.Module):
    def __init__(self,
                 num_experts: int,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 rank: int = 0):
        super(Expert, self).__init__()
        self.w_1 = FMoELinear(num_experts, idim, hidden_units, bias=True, rank=rank)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = FMoELinear(num_experts, hidden_units, idim, bias=True, rank=rank)

    def forward(self,
                xs: torch.Tensor,
                fwd_expert_count: torch.Tensor,
                capacity: float = -1.0) -> torch.Tensor:
        h = self.w_1(xs, fwd_expert_count, capacity=capacity)
        h = self.dropout(self.activation(h))
        h = self.w_2(h, fwd_expert_count, capacity=capacity)
        return h


class FmoeCatEmbedFeedForward(torch.nn.Module):
    def __init__(self, idim, embed_dim, num_experts=4, rank=0, world_size=1, hidden_units=1024,
                 dropout_rate=0.0, activation=torch.nn.ReLU(), capacity_factor=-1.0,
                 router_regularization="l1_plus_importance", router_with_bias=False,
                 keep_expert_output=False, rand_init_router=False, comm=None):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router_regularization = router_regularization
        # expert network
        self.experts = Expert(num_experts, idim, hidden_units, dropout_rate,
                              activation=activation, rank=rank)
        # router
        router_input_dim = idim + embed_dim
        self.router_weights = torch.nn.Parameter(torch.zeros(router_input_dim, num_experts * world_size))
        if router_with_bias:
            self.router_bias = torch.nn.Parameter(torch.zeros(num_experts * world_size))
        else:
            self.router_bias = None
        self.sparseLoss = SparseL1Loss(world_size)
        self.balanceLoss = BalanceImportanceLoss(world_size)
        if rand_init_router:
            torch.nn.init.xavier_uniform_(self.router_weights, gain=0.5)
        self.keep_expert_output = keep_expert_output
        mark_module_parallel_comm(self.experts, "mp")
        setattr(self.router_weights, "dp_comm", "dp_mean")
        if router_with_bias:
            setattr(self.router_bias, "dp_comm", "dp_mean")

    def gate(self, inputs):
        router_logits = torch.einsum('ij,jk->ik', [inputs, self.router_weights])
        if self.router_bias is not None:
            router_logits = router_logits + self.router_bias
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        gate_value, gate_idx = router_probs.max(dim=-1)
        all_samples = router_probs.size(0)
        # router regularization
        if self.router_regularization == "l1_plus_importance":
            l1_loss, l1_loss_item, n_samples = self.sparseLoss(router_probs, group=self.comm)
            all_samples = n_samples
            importance_loss, importance_loss_item = self.balanceLoss(router_probs, group=self.comm)
            aux_loss = ((l1_loss, l1_loss_item), (importance_loss, importance_loss_item))
        else:
            raise NotImplementedError("Not supported router regularization type: {}".format(
                                      self.router_regularization))
        return gate_idx, gate_value, aux_loss, all_samples

    def forward(self, inputs, embed):
        assert inputs.dim() == 3
        batch_size, max_steps, input_dim = inputs.size()
        inputs = inputs.view(-1, input_dim)
        embed_dim = embed.size(-1)
        embed = embed.view(-1, embed_dim)
        router_inputs = torch.cat([embed, inputs], dim=-1)
        gate_idx, gate_value, aux_loss, all_samples = self.gate(router_inputs)
        all_experts = self.num_experts * self.world_size
        capacity = int(self.capacity_factor * all_samples / all_experts)
        expert_outputs = _fmoe_general_global_forward(
                inputs, gate_idx, self.experts, self.num_experts,
                self.world_size, capacity=capacity, comm=self.comm)
        if not self.keep_expert_output:
            expert_outputs = expert_outputs * gate_value.unsqueeze(1)
        output = expert_outputs.view(batch_size, max_steps, -1)
        return output, aux_loss
