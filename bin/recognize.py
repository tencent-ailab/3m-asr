import time
import argparse
import yaml
import importlib

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.file_utils import read_symbol_table, read_non_lang_symbols, load_json_cmvn
from utils.common import set_conf


def read_char_dict(file_path):
    char_dict = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    return char_dict


def main(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    job_type = configs['job_type']
    model_conf = configs['model_conf']
    data_conf = configs['dataset_conf']
    if job_type == 'moe':
        set_conf(model_conf, 'moe_conf', 'rank', rank)
        set_conf(model_conf, 'moe_conf', 'world_size', world_size)
    # build loader
    dataset_proto = configs.get('dataset_proto')
    dataset_module = importlib.import_module('dataset.' + dataset_proto)
    Dataset = dataset_module.Dataset
    input_dim = data_conf['fbank_conf']['num_mel_bins']
    data_conf['filter_conf']['max_length'] = 102400
    data_conf['filter_conf']['min_length'] = 10
    data_conf['filter_conf']['token_max_length'] = 102400
    data_conf['filter_conf']['token_min_length'] = 0
    data_conf['speed_perturb'] = False
    data_conf['spec_aug'] = False
    data_conf['shuffle'] = False
    data_conf['sort'] = False
    data_conf['fbank_conf']['dither'] = 0.0
    data_conf['batch_conf']['batch_type'] = 'static'
    data_conf['batch_conf']['batch_size'] = 1
    bpe_model = args.bpe_model
    symbol_table = read_symbol_table(args.symbol_table)
    output_dim = len(symbol_table.keys())
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)
    test_dataset = Dataset(args.data_type, args.test_data, symbol_table,
            data_conf, bpe_model, non_lang_syms, partition=False)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)
    char_dict = read_char_dict(args.symbol_table)
    cmvn = None
    if args.cmvn_file is not None:
        cmvn = load_json_cmvn(args.cmvn_file)
    # build model
    nnet_proto = configs.get('nnet_proto')
    nnet_module = importlib.import_module("model." + nnet_proto)
    model = nnet_module.Net(input_dim, output_dim, **model_conf)
    param_dict = torch.load(args.load_path, map_location='cpu')
    if job_type == 'moe':
        model.load_state_dict_comm(param_dict)
    else:
        model.load_state_dict(param_dict)
    if torch.cuda.is_available() and args.cuda:
        model = model.cuda()
    # inference
    model.eval()
    start_time = time.time()
    beam_size = args.beam_size
    ctc_weight = args.ctc_weight
    reverse_weight = args.reverse_weight
    if args.local_rank == 0:
        wf = open(args.output_file, 'w')
    with torch.no_grad():
        for _, batch_data in enumerate(test_data_loader):
            keys_batch, feat, target, lens, label_lens = batch_data
            key_name = keys_batch[0]
            feat_len = torch.tensor([feat.size(1)]).int()
            if cmvn is not None:
                feat = (feat - cmvn[0]) * cmvn[1]
            if torch.cuda.is_available() and args.cuda:
                feat = feat.cuda()
                feat_len = feat_len.cuda()
            if args.mode == 'ctc_greedy_search':
                hyps = model.ctc_greedy_search(feat, feat_len)
                hyp = hyps[0]
            elif args.mode == 'ctc_prefix_beam_search':
                hyps, _ = model.ctc_prefix_beam_search(feat, feat_len, beam_size)
                hyp = hyps[0][0]
            elif args.mode == 'attention_rescoring':
                hyp = model.attention_rescoring(feat, feat_len, beam_size,
                        ctc_weight=ctc_weight, reverse_weight=reverse_weight)
            else:
                print("decode mode {} not Implemented".format(args.mode))
                raise NotImplementedError
            content = ''
            for w in hyp:
                content += char_dict[w]
            if args.local_rank == 0:
                print("{} {}".format(key_name, content))
                wf.write("{} {}\n".format(key_name, content))
    duration = time.time() - start_time
    if args.local_rank == 0:
        wf.close()
        print("decode cost {} seconds".format(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch ASR --- inference to get AM score")
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--load_path', required=True, help='load path')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search',
                            'ctc_prefix_beam_search',
                            'attention_rescoring'
                        ], default='attention_rescoring', help='decode mode')
    parser.add_argument('--beam_size', type=int, default=10, help='beam size')
    parser.add_argument('--ctc_weight', type=float, default=0.5, help='ctc weight')
    parser.add_argument('--reverse_weight', type=float, default=0.0, help='reverse weight')

    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument('--bpe_model', default=None,
                        help='bpe model')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--cmvn_file', type=str, default=None, help='cmvn json file')

    parser.add_argument('--local_rank', type=int, help='local process ID for parallel traininig')
    args = parser.parse_args()
    assert torch.cuda.is_available() and args.cuda
    main(args)
