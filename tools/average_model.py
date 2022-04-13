import argparse
import torch

parser = argparse.ArgumentParser("Average model parameters")
parser.add_argument("--model_list", required=True, help="list of model to be averaged")
parser.add_argument("--output_file", required=True, help="output model file")
args = parser.parse_args()

model_list = args.model_list
output_file = args.output_file

with open(model_list, 'r') as f:
    model_files = [line.strip() for line in f.readlines()]
count = len(model_files)

state_dict = {}
for i, model_file in enumerate(model_files):
    model_dict = torch.load(model_file, map_location='cpu')
    for key, param in model_dict.items():
        if i == 0:
            state_dict[key] = param
        else:
            assert key in state_dict
            state_dict[key] += param
for k in state_dict.keys():
    if state_dict[k] is not None:
        state_dict[k] = torch.true_divide(state_dict[k], count)
torch.save(state_dict, output_file)
