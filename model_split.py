#!/usr/bin/env python3
#coding:utf-8

import argparse
import os
import re
import json
import time
import utils
import config
import logging
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import make_model, Transformer_encode, Transformer_decode

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    pass
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    
    encoder = model.encoder
    decoder = model.decoder
    src_embed = model.src_embed
    tgt_embed = model.tgt_embed
    generator = model.generator
    
    model_encode = Transformer_encode(encoder, src_embed)
    model_decode = Transformer_decode(decoder, tgt_embed)

    print('saving models...')
    #model_path = './experiment/model.pth'
    #torch.save(model.state_dict(), config.model_path)
    outpath = './experiment/'

    model_path = os.path.join(outpath, 'model_all.pth')
    encode_path = os.path.join(outpath, 'encoder_all.pth')
    decoder_path = os.path.join(outpath, 'decoder_all.pth')
    generator_path = os.path.join(outpath, 'generator_all.pth')

    torch.save(model, model_path)
    torch.save(model_encode, encode_path)
    torch.save(model_decode, decoder_path)
    torch.save(generator, generator_path)
    print('models saved...')

    
