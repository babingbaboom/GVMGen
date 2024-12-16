# .th -> state_dict.bin
import os
import torch
from module.decoder.utils import export
import argparse

def load_checkpoint(checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    print(checkpoint_dict)

def main():
    parser = argparse.ArgumentParser(description='Script for processing video and model paths.')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the model checkpoint.')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to the output.')
    args = parser.parse_args()
    
    export.export_lm(args.checkpoint_path, args.output_path)
