#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import re
import json
import logging
#from flask import Flask, request, render_template, jsonify, json

import time
import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
#from train import translate
from model import make_model

from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

nvmlInit() #初始化

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT No Process Batch Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--logfile', type=str, default="report.txt", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    logfile = args.logfile
    batch_size = args.batch_size

    # 开始计时
    start = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # 加载数据文件
    txts = readtxt(datafile)
    if txts:
        sentences = txts.splitlines()
        sentences = list(filter(None, sentences))
        total_sent = len(sentences)
    else:
        print('data file error')
        sys.exit()
    

    print(' NMT Task: No Process Batch '.center(40,'-'))
    print('total sentences:%d'%total_sent)
    print('Building model...')
    # 加载模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    loadstime = (time.time() - start)*1000
    memory = GPU_memory(0)

    # 开始计时
    # start = time.time()
    print('create task...')
    if batch_size==1:
        translate_texts(sentences, model, beam_search=True)
    else:
        result = translate_batch(sentences, model, batch_size=batch_size)

    predict_time = (time.time() - start)*1000
    avetime = predict_time/total_sent
    
    print('加载模型用时:%.3f 毫秒' % loadstime )
    print('Used Memory:%d MB' % memory)
    print('预测总计用时:%.3f 毫秒' % predict_time )
    print('预测单句用时:%.3f 毫秒' % avetime )

    result = 'NoProcessBatch', total_sent, memory, loadstime, predict_time, avetime, batch_size
    # 追加到日志文件
    savetofile(result, logfile)
