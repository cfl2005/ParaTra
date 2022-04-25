#!/usr/bin/env python3
#coding:utf-8

import argparse
import os
import re
import sys
import json
import logging
from math import *
#from flask import Flask, request, render_template, jsonify, json
#import multiprocessing
import time
import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from train import translate
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp

from translate import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')

nvmlInit()

from Pytorch_Memory_Utils.gpu_mem_track import MemTracker
gpu_tracker = MemTracker()


def translate_batch_sub(sentences, batch_size=64):
    pass
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    #model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    translate_batch(sentences, model, batch_size=batch_size)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model Single Process Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--logfile', type=str, default="report.txt", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    logfile = args.logfile
    batch_size = args.batch_size

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    mp.set_start_method('spawn')

    txts = readtxt(datafile)
    if txts:
        sentences = txts.splitlines()
        sentences = list(filter(None, sentences))
        total_sent = len(sentences)
    else:
        print('data file error')
        sys.exit()

    obj = GPU_MEM()
    obj.build()
    obj.start()

    print(' NMT Task: Single Process Batch '.center(40,'-'))
    print('total sentences:%d'%total_sent)
    print('Building model...')
    start = time.time()

    gpu_tracker.track()                
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    
    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    loadstime = (time.time() - start)*1000
    gpu_tracker.track()               

    print('Create process...')
    start = time.time()
    torch.cuda.empty_cache()
    p = mp.Process(target=translate_batch, args=(sentences, model, True, batch_size))
   
    # p = mp.Process(target=translate_batch_sub, args=(sentences, batch_size))
    gpu_tracker.track()                
    
    print('Start process...')
    p.start()
    memory = GPU_memory(0)
    print('Process running...')
    p.join()
    gpu_tracker.track()              

    obj.stop()
    print('mem data:', obj.data)
    print('mem_ave:', obj.mem_ave())
    print('mem_max:', obj.mem_max())

    predict_time = (time.time() - start)*1000
    avetime = predict_time/total_sent
    

    result = {'name':'SingleProcess', 'total_sent':total_sent,
                'memory':memory, 'loadstime': loadstime, 
                'predict_time':predict_time, 'avetime':avetime, 'batch_size':batch_size,
                'mem data':obj.data, 'mem_ave': obj.mem_ave(), 'mem_max':obj.mem_max()
             }

    savetofile(json.dumps(result), logfile)
