import os
import sys
import time
import random
import yaml
import copy
import argparse
import traceback
import importlib
import numpy as np
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.distributed import DDPWrapper as DDP
from utils.distributed import DistributedGroupedDataParallel as DGDP
from utils.fmoe_localComm_bmuf import prepare_moe_info
from utils.bmuf import BmufTrainer
from utils.fmoe_localComm_bmuf import BmufTrainer as MoeBmufTrainer

from utils.lr_scheduler import build_optimizer
from utils.logger import set_logger
from utils.common import set_conf
from loss.loss_compute import MetricStat
from utils.file_utils import read_symbol_table, read_non_lang_symbols, load_json_cmvn
from dataset.utils import DistributedEvenLoader

cudnn.benchmark = False
cudnn.deterministic = True
MASTER_NODE = 0


class Trainer(object):
    def __init__(self, args, cfg):
        self.output_dir = args.output_dir
        self.local_rank = args.local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.cfg = cfg
        # set log
        logger_name = "logger.{}".format(self.rank)
        log_file = args.log_file.replace("WORKER-ID", str(self.rank))
        self.log_f = set_logger(logger_name, log_file)
        # check the job and sync method
        self.check_job_sync_method()
        # cmvn stat
        self.cmvn = None
        if args.cmvn_file is not None:
            self.cmvn = load_json_cmvn(args.cmvn_file)
        # build loader
        self.build_loader(args)
        # build model
        self.build_model(args)
        # summary writer
        writer_dir = os.path.join(self.output_dir, "summary/rank%d" % self.rank)
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=writer_dir)
        # metric stat
        metric_tags = self.model.metric_tags
        self.train_metric = MetricStat(metric_tags)
        self.valid_metric = MetricStat(metric_tags)
        # build optimizer
        self.build_optimizer()
        # check checkpoint
        self.build_checkpoint()

        self.stop_step = 0
        self.num_trained = 0

    def check_job_sync_method(self):
        # job type, dense or moe
        self.job_type = self.cfg['job_type']
        assert self.job_type in ['dense', 'moe']
        if self.job_type == 'moe':
            nnet_proto = self.cfg['nnet_proto']
            # make sure model proto is moe style
            assert 'moe' in nnet_proto
        # sync method, ddp or bmuf 
        self.sync_method = self.cfg['train_conf']['sync_method']
        assert self.sync_method in ['ddp', 'bmuf']

    def build_loader(self, args):
        dataset_proto = self.cfg['dataset_proto']
        dataset_module = importlib.import_module('dataset.' + dataset_proto)
        Dataset = dataset_module.Dataset
        train_data_conf = self.cfg['dataset_conf']
        input_dim = train_data_conf['fbank_conf']['num_mel_bins']
        cv_data_conf = copy.deepcopy(train_data_conf)
        cv_data_conf['speed_perturb'] = False
        cv_data_conf['spec_aug'] = False
        cv_data_conf['shuffle'] = False

        data_type = args.data_type
        symbol_table = read_symbol_table(args.symbol_table)
        non_lang_syms = read_non_lang_symbols(args.non_lang_syms)
        bpe_model = args.bpe_model  # default None here
        self.train_dataset = Dataset(data_type, args.train_data, symbol_table,
                                train_data_conf, bpe_model, non_lang_syms, partition=True)
        # no partition for cv data list
        # each worker have the same data list
        self.cv_dataset = Dataset(data_type, args.cv_data, symbol_table,
                             cv_data_conf, bpe_model, non_lang_syms, partition=False)

        self.train_data_loader = DistributedEvenLoader(
                                       self.local_rank,
                                       self.train_dataset,
                                       batch_size=None,
                                       pin_memory=args.pin_memory,
                                       num_workers=args.num_workers,
                                       prefetch_factor=args.prefetch)
        self.cv_data_loader = DataLoader(self.cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
        self.input_dim = input_dim
        self.output_dim = len(symbol_table.keys())
        self.log_f.info("input_dim: {}, output_dim: {}".format(self.input_dim, self.output_dim))

    def build_model(self, args):
        nnet_proto = self.cfg['nnet_proto']
        nnet_module = importlib.import_module("model." + nnet_proto)
        model_conf = self.cfg['model_conf']
        # set `rank` and `world_size` under moe_conf
        if self.job_type == 'moe':
            expert_rank, expert_world_size, mp_group, dp_group = \
                    prepare_moe_info()
            self.expert_rank = expert_rank
            self.expert_world_size = expert_world_size
            set_conf(model_conf, 'moe_conf', 'rank', self.expert_rank)
            set_conf(model_conf, 'moe_conf', 'world_size', self.expert_world_size)
            set_conf(model_conf, 'moe_conf', 'comm', mp_group)
        self.model = nnet_module.Net(self.input_dim,
            self.output_dim, **model_conf)
        # init model
        if self.job_type == 'moe':
            if args.init_embed_model is not None:
                self.model.init_embed_model(args.init_embed_model)
                self.log_f.info("Initialize embedding model from: {}" \
                                .format(args.init_embed_model))
            if args.init_experts_from_base is not None:
                self.model.init_experts_from_base(
                        args.init_experts_from_base)
                self.log_f.info("Initialize experts model from baseline model: {}" \
                                .format(args.init_experts_from_base))
        if args.init_model is None:
            self.log_f.info("Random initialize model")
        else:
            param_dict = torch.load(args.init_model, map_location='cpu')
            if self.job_type == 'moe':
                self.model.load_state_dict_comm(param_dict)
            else:
                self.model.load_state_dict(param_dict)
            self.log_f.info("Initialize model from: {}" \
                            .format(args.init_model))
        num_param = 0
        for param in self.model.parameters():
            num_param += param.numel()
        self.log_f.info("model proto: {},\tmodel_size: {} M".format(
            nnet_proto, num_param / 1000 / 1000))
        # place on gpu device
        self.model.cuda(self.local_rank)
        # sync method
        if self.sync_method == 'bmuf':
            train_conf = self.cfg['train_conf']
            block_momentum = train_conf.get('block_momentum', 0.9)
            block_lr = train_conf.get('block_lr', 1.0)
            if self.job_type == 'moe':
                self.bmuf_trainer = MoeBmufTrainer(
                        self.model, dp_group, mp_group,
                        self.expert_rank, MASTER_NODE,
                        block_momentum, block_lr)
            else:
                self.bmuf_trainer = BmufTrainer(
                        MASTER_NODE, self.model,
                        block_momentum, block_lr)
        else:
            if self.job_type == 'moe':
                self.model = DGDP(self.model, dp_group,
                        self.expert_rank, MASTER_NODE)
            else:
                self.model = DDP(self.model, device_ids=[self.local_rank],
                        find_unused_parameters=True)

    def build_optimizer(self):
        train_conf = self.cfg['train_conf']
        named_params = self.model.named_parameters()
        lr = train_conf.get('lr', 1e-4)
        optim_type = train_conf.get('optim')
        optim_conf = train_conf.get('optim_conf', {})
        schedule_type = train_conf.get('schedule_type')
        schedule_conf = train_conf.get('schedule_conf', {})
        logger = self.log_f
        if 'name_nodecay' not in schedule_conf:
            schedule_conf['name_nodecay'] = [".bias", "Norm.weight"]
        self.optimizer = build_optimizer(named_params, schedule_type, schedule_conf,
                                         lr, optim_type, optim_conf, logger)
        self.log_f.info("[Optimizer] scheduler wrapper is {}".format(
            self.optimizer.__class__.__name__))

    def build_checkpoint(self):
        chkpt_fn = "{}/chkpt".format(self.output_dir)
        if os.path.isfile(chkpt_fn):
            chkpt = torch.load(chkpt_fn, map_location="cpu")
            self.start_epoch = chkpt["epoch"]
            self.best_model = chkpt["best_model"]
            self.global_step = chkpt["global_step"]
            self.best_valid_loss = chkpt["best_valid_loss"]
            self.recent_models = chkpt["recent_models"]
            self.optimizer.load_state_dict(chkpt["optim"])
            cur_lr = self.optimizer.get_learning_rate()
            self.optimizer.adjust_learning_rate(cur_lr)
            # load most recent model
            param_dict = torch.load(self.recent_models[-1], map_location="cpu")
            if self.job_type == 'moe':
                self.model.load_state_dict_comm(param_dict)
            else:
                self.model.load_state_dict(param_dict)
            # sync bmuf parameters
            if self.sync_method == 'bmuf':
                self.bmuf_trainer.reset_param_vectors()
                if self.job_type == 'moe':
                    self.bmuf_trainer.load_state_dict_comm(chkpt['bmuf'])
                else:
                    self.bmuf_trainer.load_state_dict(chkpt['bmuf'])
            self.log_f.info("loading checkpoint {} to continue training. "
                            "current lr is {}".format(chkpt_fn, cur_lr))
            self.log_f.info("loading most recent model from {}".format(
                            self.recent_models[-1]))
        else:
            self.log_f.info("no checkpoint, start training from scratch")
            self.start_epoch = 1
            self.best_model = "{}/model.epoch-0.step-0".format(self.output_dir)
            self.recent_models = [self.best_model]
            self.global_step = 0

            self.best_valid_loss = float('inf')
            # only maintain single best_model, saved by MASTER_NDOE
            if self.job_type == 'moe':
                model_state_dict = self.model.state_dict_comm()
            else:
                model_state_dict = self.model.state_dict()
            if self.rank == MASTER_NODE:
                torch.save(model_state_dict, self.best_model)
            # save chkpt,  only master_node will save the file
            self.save_chkpt(self.start_epoch)

    def save_chkpt(self, epoch):
        chkpt_fn = "{}/chkpt".format(self.output_dir)
        optim_state = self.optimizer.state_dict()
        chkpt = {'epoch': epoch,
                 'best_model': self.best_model,
                 'best_valid_loss': self.best_valid_loss,
                 'recent_models': self.recent_models,
                 'global_step': self.global_step,
                 'optim': optim_state}
        # bmuf state
        if self.sync_method == 'bmuf':
            if self.job_type == 'moe':
                bmuf_state = self.bmuf_trainer.state_dict_comm()
            else:
                bmuf_state = self.bmuf_trainer.state_dict()
            chkpt['bmuf'] = bmuf_state
        if self.rank == MASTER_NODE:
            torch.save(chkpt, chkpt_fn)

    def save_model_state(self, epoch):
        cur_model = "{}/model.epoch-{}.step-{}".format(
                self.output_dir, epoch, self.global_step)
        if self.job_type == 'moe':
            model_state_dict = self.model.state_dict_comm()
        else:
            model_state_dict = self.model.state_dict()
        if self.rank == MASTER_NODE:
            torch.save(model_state_dict, cur_model)
        self.recent_models += [cur_model]
        num_recent_models = self.cfg['train_conf'].get('num_recent_models', -1)
        if num_recent_models > 0 and len(self.recent_models) > num_recent_models:
            pop_model = self.recent_models.pop(0)
            if self.rank == MASTER_NODE:
                os.remove(pop_model)

    def should_early_stop(self):
        early_stop_count = self.cfg['train_conf'].get('early_stop_count', 10)
        return self.stop_step >= early_stop_count

    def train_one_epoch(self, epoch):
        cur_lr = self.optimizer.get_learning_rate()
        self.log_f.info("Epoch {} start, lr {}".format(epoch, cur_lr))
        train_conf = self.cfg['train_conf']
        # by sentences
        log_period = train_conf.get('log_period', 1000)
        valid_period = train_conf.get('valid_period', -1)
        schedule_type = train_conf['schedule_type']
        # by global step
        accum_grad = train_conf.get('accum_grad', 1)
        if self.sync_method == 'bmuf':
            sync_period = train_conf.get('sync_period', 10)

        frames_total = 0
        frames_log = 0
        start_time = time.time()
        epoch_start_time = start_time
        # train mode
        self.model.train()
        # run data
        # DistributedEvenLoader ensure even data list
        for batch_idx, batch_data in enumerate(self.train_data_loader):
            keys, data, target, lens, label_lens = batch_data
            if self.cmvn is not None:
                data = (data - self.cmvn[0]) * self.cmvn[1]
            # put data on corresponding GPU
            data = data.cuda(self.local_rank)
            target = target.cuda(self.local_rank)
            lens = lens.cuda(self.local_rank)
            label_lens = label_lens.cuda(self.local_rank)

            batch_size = data.size(0)
            self.global_step += 1
            if self.sync_method == 'ddp' and self.job_type == 'dense' \
                    and self.global_step % accum_grad != 0:
                # not synchronize gradient
                context = self.model.no_sync
            else:
                context = nullcontext
            with context():
                res = self.model(data, lens, target, label_lens)
                loss, metrics, counts = self.model.cal_loss(
                        res, target, label_lens)
                loss = loss / accum_grad
                loss.backward()
            self.train_metric.update_stat(metrics, counts)
            if schedule_type in ["warmup_linear", "warmup_cosine", "warmup_plateau", "warmup_noam"]:
                if self.global_step % accum_grad == 0:
                    self.optimizer.addStep_adjustLR(1)
            elif schedule_type == "period_adjust":
                self.optimizer.addStep_adjustLR(batch_size)
            if self.global_step % accum_grad == 0:
                # self-defined ddp for moe 
                if self.job_type == 'moe' and self.sync_method == 'ddp':
                    self.model.allreduce_grad()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # sync for bmuf
            if self.sync_method == 'bmuf' and self.global_step % sync_period == 0:
                self.update_and_sync()
            frames = torch.sum(lens).item()
            frames_log += frames
            self.num_trained += batch_size
            # log info
            if self.num_trained % log_period < batch_size:
                log_time = time.time()
                elapsed = log_time - start_time
                avg_stat = self.train_metric.log_stat()
                avg_str = []
                for tag, stat in zip(self.train_metric.tags, avg_stat):
                    self.writer.add_scalar("train/%s"%tag, stat, self.global_step)
                    avg_str += ["{}: {:.6f},".format(tag, stat)]
                avg_str = '\t'.join(avg_str)
                cur_lr = self.optimizer.get_learning_rate()
                self.log_f.info("Epoch: {},\tTrained sentences: {},\t"
                                "{}\tlr: {:.8f},\tfps: {:.1f} k".format(epoch,
                                self.num_trained, avg_str, cur_lr, frames_log/elapsed/1000))
                start_time = log_time
                frames_total += frames_log
                frames_log = 0
            # validation and save model
            if valid_period > 0 and self.num_trained % valid_period < batch_size:
                # sync before valid
                if self.sync_method == 'bmuf':
                    self.update_and_sync()
                self.valid(epoch)
                self.save_chkpt(epoch)
        frames_total += frames_log
        train_stat = self.train_metric.summary_stat()
        self.train_metric.reset()
        avg_str = []
        for tag, stat in zip(self.train_metric.tags, train_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        elapsed = time.time() - epoch_start_time
        self.log_f.info("Epoch {} Done,\t{}\tAvg fps: {:.1f} k,"
                        "\tTime: {:.1f} hr,\t# frames: {:.1f} M".format(
                        epoch, avg_str, frames_total/elapsed/1000,
                        elapsed/3600, frames_total/1000/1000))
        # validation after one epoch
        if self.sync_method == 'bmuf':
            self.update_and_sync()
        self.valid(epoch)
        self.save_chkpt(epoch + 1)

    def valid(self, epoch):
        self.log_f.info("Start validation")
        log_period = 200
        num_sentences = 0
        self.model.eval()

        frames_total = 0
        frames_log = 0
        start_time = time.time()
        valid_start_time = start_time
        schedule_type = self.cfg['train_conf']['schedule_type']
        for batch_idx, batch_data in enumerate(self.cv_data_loader):
            key, data, target, lens, label_lens = batch_data
            if self.cmvn is not None:
                data = (data - self.cmvn[0]) * self.cmvn[1]
            # put data on corresponding GPU device
            data = data.cuda(self.local_rank)
            target = target.cuda(self.local_rank)
            lens = lens.cuda(self.local_rank)
            label_lens = label_lens.cuda(self.local_rank)
            batch_size = data.size(0)
            with torch.no_grad():
                res = self.model(data, lens, target, label_lens)
                loss, metrics, counts = self.model.cal_loss(
                    res, target, label_lens)
            self.valid_metric.update_stat(metrics, counts)
            frames = torch.sum(lens).item()
            frames_log += frames
            num_sentences += batch_size
            if num_sentences % log_period < batch_size:
                log_time = time.time()
                elapsed = log_time - start_time
                avg_stat = self.valid_metric.log_stat()
                avg_str = []
                for tag, stat in zip(self.valid_metric.tags, avg_stat):
                    avg_str += ["{}: {:.6f},".format(tag, stat)]
                avg_str = '\t'.join(avg_str)
                self.log_f.info("Valided Sentences: {},\t{}\t"
                                "fps: {:.1f} k".format(num_sentences,
                                avg_str, frames_log/elapsed/1000))
                frames_total += frames_log
                frames_log = 0
                start_time = log_time
        # finish validation
        frames_total += frames_log
        valid_stat = self.valid_metric.summary_stat()
        elapsed = time.time() - valid_start_time
        avg_str = []
        for tag, stat in zip(self.valid_metric.tags, valid_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        self.log_f.info("Validation Done,\t{}\tAvg fps: {:.1f} k,"
                        "\tTime: {} s\t# frames: {:.1f}M".format(
                        avg_str, frames_total/elapsed/1000, elapsed,
                        frames_total/1000/1000))
        # sync validation results
        tot_sum = self.valid_metric.total_sum
        tot_num = self.valid_metric.total_count
        loss_tensor = torch.FloatTensor([tot_sum, tot_num])
        loss_tensor = loss_tensor.cuda(self.local_rank)
        dist.all_reduce(tensor=loss_tensor, async_op=False)
        self.valid_metric.reset()
        reduced_stat = loss_tensor[0] / loss_tensor[1]
        valid_stat = reduced_stat.cpu().numpy()
        self.log_f.info("reduced valid loss: {}".format(valid_stat[0]))
        if self.rank == MASTER_NODE:
            for tag, stat in zip(self.valid_metric.tags, valid_stat):
                self.writer.add_scalar("valid/%s"%tag, stat, self.global_step)
        # save model state
        self.save_model_state(epoch)
        # check best loss
        valid_loss = valid_stat[0]
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_model = "{}/best_valid_model".format(self.output_dir)
            if self.rank == MASTER_NODE:
                os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
            self.log_f.info("new best_valid_loss: {}, storing best model: {}".format(
                            self.best_valid_loss, self.recent_models[-1]))
            self.stop_step = 0
            if schedule_type == "cv_adjust":
                self.optimizer.reset_step()
        else:
            self.stop_step += 1
            if schedule_type == "cv_adjust":
                self.optimizer.addStep_adjustLR(1)
        # back to train mode
        self.model.train()

    def update_and_sync(self):
        # only used in bmuf
        if not self.bmuf_trainer.update_and_sync():
            # model diverge
            self.log_f.warning("Model Diverges!")
            self.log_f.info("Reload {} and decay the "
                            "learning rate".format(self.best_model))
            # load parameter on cpu first
            param_dict = torch.load(self.best_model, map_location='cpu')
            if self.job_type == 'moe':
                self.model.load_state_dict_comm(param_dict)
            else:
                self.model.load_state_dict(param_dict)
            self.optimizer.half_learning_rate()
            self.stop_step += 1

    def run(self):
        max_epochs = self.cfg['train_conf']['max_epochs']
        self.log_f.info("Start training")
        try:
            for epoch in range(self.start_epoch, max_epochs + 1):
                if self.should_early_stop():
                    self.log_f.info("Early stopping")
                    break
                self.train_dataset.set_epoch(epoch)
                self.train_data_loader.reset_flag()
                self.train_one_epoch(epoch)
            self.log_f.info("Training Finished")
            if self.rank == MASTER_NODE:
                os.system("ln -s {} {}/final.nnet".format(
                    os.path.abspath(self.best_model), self.output_dir))
        except Exception as e:
            self.log_f.error("training exception: %s" % e)
            traceback.print_exc()


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    # load config
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    # init distributed method
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    train_conf = configs['train_conf']
    seed = train_conf.get('seed', 777)
    init_seed(seed + rank)
    # set default device
    torch.cuda.set_device(args.local_rank)
    trainer = Trainer(args, configs)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch ASR training')

    parser.add_argument('--output_dir', required=True, type=str,
            help='path to save the final model')
    parser.add_argument('--train_data', required=True, type=str,
            help='train data list')
    parser.add_argument('--cv_data', required=True, type=str,
            help='cv data list')
    parser.add_argument('--log_file', required=True, type=str,
            help='log file')
    parser.add_argument('--config', required=True, type=str,
            help='training yaml config file')
    parser.add_argument('--local_rank', type=int,
            help='local process ID for parallel training')

    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument('--non_lang_syms',
                        help='non-linguistic symbol file. One symbol per line.')
    parser.add_argument('--bpe_model', default=None,
                        help='bpe model')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--cmvn_file', type=str, default=None, help='cmvn json file')

    parser.add_argument('--init_model', type=str, default=None, help='initial model')
    parser.add_argument('--init_embed_model', type=str, default=None,
                        help='initial embedding model')
    parser.add_argument('--init_experts_from_base', type=str, default=None,
                        help='initialize MoE model from base model')

    args = parser.parse_args()
    main(args)
