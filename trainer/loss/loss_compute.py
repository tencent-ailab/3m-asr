import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class MetricStat(object):
    """
    Metric statistics class
    Args:
        - tags: name tag for each metric
    """
    def __init__(self, tags):
        super(MetricStat, self).__init__()
        self.tags = tags
        self.total_count = [0 for t in tags]
        self.total_sum = [0.0 for t in tags]
        self.log_count = [0 for t in tags]
        self.log_sum = [0.0 for t in tags]

    def update_stat(self, metrics, counts):
        for i, (m, c) in enumerate(zip(metrics, counts)):
            self.log_count[i] += c
            self.log_sum[i] += m

    def log_stat(self):
        """get recent average statistics"""
        avg = []
        for i, (m, c) in enumerate(zip(self.log_sum, self.log_count)):
            avg_stat = 0.0 if c == 0 else m / c
            avg += [avg_stat]
            self.total_sum[i] += m
            self.log_sum[i] = 0.0
            self.total_count[i] += c
            self.log_count[i] = 0
        return avg

    def summary_stat(self):
        """get total average statistics"""
        avg = []
        for i in range(len(self.tags)):
            self.total_sum[i] += self.log_sum[i]
            self.total_count[i] += self.log_count[i]
            avg_stat = 0.0
            if self.total_count[i] != 0:
                avg_stat = self.total_sum[i] / self.total_count[i]
            avg += [avg_stat]
        return avg

    def reset(self):
        for i in range(len(self.tags)):
            self.total_sum[i] = 0.0
            self.total_count[i] = 0
            self.log_sum[i] = 0.0
            self.log_count[i] = 0


class CELoss(nn.Module):
    def __init__(self, padding_idx, blank_idx, mean_in_frames=False):
        super(CELoss, self).__init__()
        self.padding_idx = padding_idx
        self.blank_idx = blank_idx
        self.mean_in_frames = mean_in_frames
        self.ce_criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    def forward(self, output, target):
        flat_out = output.view(-1, output.size(2))
        prob = F.softmax(flat_out, dim=-1)
        log_prob = F.log_softmax(flat_out, dim=-1)
        target = target.view(-1)
        assert log_prob.size(0) == target.size(0)
        ce_loss = self.ce_criterion(log_prob, target)
        # likely
        mask = target.ne(self.padding_idx)
        num_classes = prob.size(1)
        frames = torch.sum(mask).item()
        true_prob = prob[mask]
        true_target = target[mask]
        likely = torch.sum(true_prob * F.one_hot(true_target, num_classes).float())
        likely = likely.item()
        # hit
        prob_max = true_prob.argmax(dim=1)
        hit = torch.sum(true_target == prob_max).item()
        # print mean of frames on ce_loss, likely and acc
        metric = (ce_loss.item(), likely, hit)
        count = (frames, frames, frames)
        # mean in frames
        if self.mean_in_frames:
            ce_loss = ce_loss / frames
        return ce_loss, metric, count


class CTCLoss(nn.Module):
    def __init__(self, blank_idx, mean_in_batch=True):
        super(CTCLoss, self).__init__()
        self.mean_in_batch = mean_in_batch
        self.blank_idx = blank_idx
        self.ctc_criterion = nn.CTCLoss(blank=blank_idx, reduction='sum',
                                        zero_infinity=True)

    def forward(self, output, lens, label, label_size):
        # output shape is [B, T, D], should transform into [T, B, D]
        log_prob = F.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
        # output loss divided by target len and mean through batch
        ctc_loss = self.ctc_criterion(log_prob, label, lens, label_size)
        loss_sum = ctc_loss.item()
        # mean through batch
        if self.mean_in_batch:
            ctc_loss = ctc_loss / lens.size(0)
        # print mean of ctc loss through utterances
        metric = (loss_sum, )
        count = (lens.size(0), )
        return ctc_loss, metric, count


class MoELayerScaleAuxLoss(nn.Module):
    def __init__(self, num_aux, aux_scale, loss_minimum):
        super(MoELayerScaleAuxLoss, self).__init__()
        assert isinstance(aux_scale, (list, tuple)) and len(aux_scale) == num_aux
        if loss_minimum is not None:
            assert isinstance(loss_minimum, (list, tuple)) and len(loss_minimum) == num_aux
        self.max_aux_scale = aux_scale
        self.aux_scale = copy.copy(aux_scale)
        self.loss_minimum = loss_minimum

    def adjust_aux_scale(self, aux_metric):
        if self.loss_minimum is None:
            return self.aux_scale
        assert len(aux_metric) == len(self.aux_scale)
        for i in range(len(aux_metric)):
            delta = (aux_metric[i] - self.loss_minimum[i]) / self.loss_minimum[i] * 3
            factor = min(delta , 1.0)
            self.aux_scale[i] = self.max_aux_scale[i] * factor
        return self.aux_scale

    def forward(self, aux_loss):
        num_aux = len(aux_loss[0])
        aux_loss_sum = [0.0 for i in range(num_aux)]
        loss = 0.0
        for i in range(len(aux_loss)):
            for j in range(num_aux):
                loss_tensor, loss_item = aux_loss[i][j][0], aux_loss[i][j][1]
                loss += self.aux_scale[j] * loss_tensor
                aux_loss_sum[j] += loss_item
        metric = tuple(aux_loss_sum)
        count = tuple([1 for j in range(num_aux)])
        return loss, metric, count


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """
    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum()
        metric = (loss.item(), )
        count = (total, )
        loss = loss / denom
        return loss, metric, count
