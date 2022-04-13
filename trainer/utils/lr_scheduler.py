import abc
import sys
import math
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

OPTIMS = {
    'sgd': optim.SGD,
    'adadelta': optim.Adadelta,
    'adam': optim.Adam
}


class OptimizerBaseWrapper(ABC):
    """
    Scheduler warpper base class of optimizer
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, max_grad_norm=-1, weight_decay=0.0, name_nodecay=None):
        self.lr_step = 0
        self.lr = lr
        self.min_lr = min_lr
        self.logger = logger
        self.max_grad_norm = max_grad_norm
        if name_nodecay is None:
            # filter trainable parameters
            params = [p for n, p in named_params if p.requires_grad]
            optimizer_group_params = [{'params': params, 'weight_decay': weight_decay}]
        else:
            logger.info("[Optimizer] noDecay_names: {}".format(name_nodecay))
            decay_params = [p for n, p in named_params if not any([nd in n for nd in name_nodecay])]
            decay_params = filter(lambda p: p.requires_grad, decay_params)
            nodecay_params = [p for n, p in named_params if any([nd in n for nd in name_nodecay])]
            nodecay_params = filter(lambda p: p.requires_grad, nodecay_params)
            optimizer_group_params = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
        # create optimizer
        if optim_type not in OPTIMS:
            raise NotImplementedError("optim_type {} not supported".format(optim_type))
        self.optimizer = OPTIMS[optim_type](optimizer_group_params, self.lr, **optim_conf)

    def get_learning_rate(self):
        return self.lr

    def adjust_learning_rate(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def half_learning_rate(self):
        self.lr *= 0.5
        self.adjust_learning_rate(self.lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # clip grad
        if self.max_grad_norm > 0:
            for group in self.optimizer.param_groups:
                params = group['params']
                clip_grad_norm_(params, self.max_grad_norm)
        # optimizer step
        self.optimizer.step()

    @abc.abstractmethod
    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        self.lr = max(self.lr, self.min_lr)
        self.adjust_learning_rate(self.lr)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['lr_step'] = self.lr_step
        state_dict['cur_lr'] = self.lr
        state_dict['wrapper_name'] = self.__class__.__name__
        return state_dict

    def load_state_dict(self, state_dict):
        self.lr_step = state_dict['lr_step']
        self.lr = state_dict['cur_lr']
        wrapper_name = state_dict['wrapper_name']
        assert wrapper_name == self.__class__.__name__
        state_dict.pop('lr_step')
        state_dict.pop('cur_lr')
        state_dict.pop('wrapper_name')
        self.optimizer.load_state_dict(state_dict)
        self.adjust_learning_rate(self.lr)


class ConstantScheduleWrapper(OptimizerBaseWrapper):
    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr < self.min_lr:
            self.lr = self.min_lr
            self.adjust_learning_rate(self.lr)


class PeriodScheduleWrapper(OptimizerBaseWrapper):
    """
    decay lr every N step
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 decay_period=10000, min_lr=1e-8, lr_decay=0.8, max_grad_norm=-1,
                 weight_decay=0.0, name_nodecay=None):
        super(PeriodScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger, min_lr=min_lr,
                max_grad_norm=max_grad_norm, weight_decay=weight_decay,
                name_nodecay=name_nodecay)
        self.lr_decay = lr_decay
        self.decay_period = decay_period

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr_step >= self.decay_period:
            self.lr = max(self.lr * self.lr_decay, self.min_lr)
            self.adjust_learning_rate(self.lr)
            self.lr_step -= self.decay_period
            self.logger.info("[Optimizer] decay lr to {}".format(self.lr))


class CVScheduleWrapper(OptimizerBaseWrapper):
    """
    decay lr if metric does not improve for continous N validation
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, lr_decay=0.5, lr_decay_count=10, max_grad_norm=-1,
                 weight_decay=0.0, name_nodecay=None):
        super(CVScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger,
                min_lr=min_lr, max_grad_norm=max_grad_norm,
                weight_decay=weight_decay, name_nodecay=name_nodecay)
        self.noImp_limit = lr_decay_count
        self.lr_decay = lr_decay

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr_step >= self.noImp_limit:
            self.lr = max(self.lr * self.lr_decay, self.min_lr)
            self.adjust_learning_rate(self.lr)
            self.logger.info("[Optimizer] decay lr to {}".format(self.lr))
            self.lr_step = 0

    def reset_step(self):
        self.lr_step = 0


class WarmupLinearScheduleWrapper(OptimizerBaseWrapper):
    """
    Linear warmup learning rate
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, warmup=0.02, total_steps=100000, max_grad_norm=-1,
                 weight_decay=0.0, name_nodecay=None):
        super(WarmupLinearScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger,
                min_lr=min_lr, max_grad_norm=max_grad_norm,
                weight_decay=weight_decay, name_nodecay=name_nodecay)
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)
        self.peak_lr = lr
        self.lr = self.min_lr

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr_step <= self.warmup_steps:
            lr = self.peak_lr * (self.lr_step / self.warmup_steps)
        else:
            lr = self.peak_lr * \
                    ((self.lr_step - self.total_steps) / (self.warmup_steps - self.total_steps))
        if self.lr_step == self.warmup_steps:
            self.logger.info("[Optimizer] Warmup to peak lr {}".format(self.peak_lr))
        self.lr = max(lr, self.min_lr)
        self.adjust_learning_rate(self.lr)

    def half_learning_rate(self):
        self.peak_lr *= 0.5


class WarmupCosineScheduleWrapper(OptimizerBaseWrapper):
    """
    linear increase lr and cosine decay lr
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, warmup=0.02, total_steps=100000, max_grad_norm=-1,
                 weight_decay=0.0, name_nodecay=None):
        super(WarmupCosineScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger,
                min_lr=min_lr, max_grad_norm=max_grad_norm,
                weight_decay=weight_decay, name_nodecay=name_nodecay)
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)
        self.peak_lr = lr
        self.lr = self.min_lr
        self.cycle = 0.5

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr_step <= self.warmup_steps:
            lr = self.peak_lr * (self.lr_step / self.warmup_steps)
        else:
            progress = (self.lr_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * 2 * self.cycle * progress))
            lr = scale * self.peak_lr
        if self.lr_step == self.warmup_steps:
            self.logger.info("[Optimizer] Warmup to peak lr {}".format(self.peak_lr))
        self.lr = max(lr, self.min_lr)
        self.adjust_learning_rate(self.lr)

    def half_learning_rate(self):
        self.peak_lr *= 0.5


class WarmupPlateauScheduleWrapper(OptimizerBaseWrapper):
    """
    linear increase lr then hold and decrease exponentially
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, t_step=1000, d_step=20000, f_step=80000,
                 max_grad_norm=-1, weight_decay=0.0, name_nodecay=None):
        super(WarmupPlateauScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger,
                min_lr=min_lr, max_grad_norm=max_grad_norm,
                weight_decay=weight_decay, name_nodecay=name_nodecay)
        assert t_step < d_step and d_step < f_step
        self.t_step = t_step
        self.d_step = d_step
        self.f_step = f_step
        self.peak_lr = lr
        self.lr = self.min_lr
        self.exponent_decay = pow(0.01, 1.0 / (f_step - d_step))

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        if self.lr_step <= self.t_step:
            lr = self.peak_lr * (self.lr_step / self.t_step)
        elif self.lr_step <= self.d_step:
            lr = self.peak_lr
        elif self.lr_step <= self.f_step:
            lr = self.lr * self.exponent_decay
        else:
            lr = self.lr
        self.lr = max(lr, self.min_lr)
        if self.lr_step == self.t_step:
            self.logger.info("[Optimizer] Warmup to peak lr {}, then hold lr".format(self.peak_lr))
        if self.lr_step == self.d_step:
            self.logger.info("[Optimizer] End holding lr, start to decay")
        if self.lr_step == self.f_step:
            self.logger.info("[Optimizer] Decay lr to {}".format(self.lr))
        self.adjust_learning_rate(self.lr)

    def half_learning_rate(self):
        self.peak_lr *= 0.5


class WarmupNoamScheduleWrapper(OptimizerBaseWrapper):
    """
    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    """
    def __init__(self, named_params, lr, optim_type, optim_conf, logger,
                 min_lr=1e-8, warmup_steps=25000, max_grad_norm=-1,
                 weight_decay=0.0, name_nodecay=None):
        super(WarmupNoamScheduleWrapper, self).__init__(
                named_params, lr, optim_type, optim_conf, logger,
                min_lr=min_lr, max_grad_norm=max_grad_norm,
                weight_decay=weight_decay, name_nodecay=name_nodecay)
        self.warmup_steps = warmup_steps
        self.peak_lr = lr
        self.lr = self.min_lr

    def addStep_adjustLR(self, delta):
        self.lr_step += delta
        lr = self.peak_lr \
             * self.warmup_steps ** 0.5 \
             * min(self.lr_step ** -0.5, self.lr_step * self.warmup_steps ** -1.5)
        if self.lr_step == self.warmup_steps:
            self.logger.info("[Optimizer] Warmup to peak lr {}".format(self.peak_lr))
        self.lr = max(lr, self.min_lr)
        self.adjust_learning_rate(self.lr)

    def half_learning_rate(self):
        self.peak_lr *= 0.5


SUPPORTED_SCHEDULER = {
    'constant': ConstantScheduleWrapper,
    'cv_adjust': CVScheduleWrapper,
    'period_adjust': PeriodScheduleWrapper,
    'warmup_linear': WarmupLinearScheduleWrapper,
    'warmup_cosine': WarmupCosineScheduleWrapper,
    'warmup_plateau': WarmupPlateauScheduleWrapper,
    'warmup_noam': WarmupNoamScheduleWrapper
}


def build_optimizer(named_params, schedule_type, schedule_conf,
                    lr, optim_type, optim_conf, logger):
    if schedule_type not in SUPPORTED_SCHEDULER:
        raise NotImplementedError("Not supported schedule type: {}".format(schedule_type))
    if optim_type not in OPTIMS:
        raise NotImplementedError("Not supported optim type: {}".format(optim_type))
    optimizer = SUPPORTED_SCHEDULER[schedule_type](
            named_params, lr, optim_type, optim_conf, logger, **schedule_conf)
    return optimizer
