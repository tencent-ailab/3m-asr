import argparse
import torch

parser = argparse.ArgumentParser("strip encoder state dict")
parser.add_argument("-i", "--input", required=True, type=str, help="input state dict")
parser.add_argument("-o", "--output", required=True, type=str, help="output state dict")
args = parser.parse_args()

model_state = torch.load(args.input, map_location="cpu")
encoder_state = {}
for k, v in model_state.items():
    if k.startswith("encoder."):
        new_k = k.replace("encoder.", "")
        encoder_state[new_k] = v
torch.save(encoder_state, args.output)
