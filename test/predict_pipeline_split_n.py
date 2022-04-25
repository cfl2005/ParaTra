#!/usr/bin/env python3
# coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import re
import time
import json
import logging
from math import *
# from flask import Flask, request, render_template, jsonify, json
import sys
import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import translate, translate_encode, translate_decode, translate_decode_split
from utils import english_tokenizer_load
from model import *

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager, Queue
from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')

from pynvml import *
nvmlInit()  # 初始化

# from Pytorch_Memory_Utils.gpu_mem_track import MemTracker  # 引用显存跟踪代码
# gpu_tracker = MemTracker()      # 创建显存检测对象


# 把文本处理后发送到队列中
def send_dat(sentences, q_text, q_result, batch_size=64, show_result=0, tmr_queue=None):
    """
    发送数据
    """
    start = time.time()
    print('sending text...')
    # 句子按batch_size拆分
    total = len(sentences)
    sentlist = [sentences[i*batch_size:(i+1)*batch_size] for i in range(ceil(total/batch_size))]
    # 发送到队列中
    for txts in sentlist:
        batch_text = get_sample(txts)
        q_text.put(batch_text)

    
    # 准备接收结果
    i = 0 
    print('receive result...')
    while 1:
        ret = q_result.get()
        if type(ret) == list:
            pass
            if show_result:
                print('result:\n', '\n'.join(ret))
            i += len(ret)
        if i >= total:
            break
    predict_time = (time.time() - start)*1000
    if not tmr_queue is None:
        tmr_queue.put(predict_time)
    print('task end...')

def proc_encode(q_text, q_enc, model_encoder=None):
    """
    编码器
    """
    # 加载模型
    if model_encoder is None:
        print('Loading model:encoder...')
        model_encoder = make_model_encode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)
        model_encoder.load_state_dict(torch.load(config.model_path_encoder))
        # model_encoder = torch.load(config.model_path_whole).to(device)
        model_encoder.eval()
    torch.cuda.empty_cache()
    
    while True:
        # print("proc_encode.")
        dat = q_text.get()
        # gpu_tracker = MemTracker() 
        # print('q_text dat:',type(dat))  #np.array(dat)
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        # gpu_tracker.track()                  # 开始检测    
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        # print('src_enc:',type(src_enc))
        # gpu_tracker.track()                  # 开始检测    
        q_enc.put( (src_enc,src_mask) )

def proc_decode(q_enc, q_result, model_decoder=None, model_generator=None):
    """
    解码器
    """
    if model_decoder is None:
        print('Loading model:decoder,generator ...')
        model_decoder, model_generator = make_model_decode(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)

        model_decoder.load_state_dict(torch.load(config.model_path_decoder))
        model_generator.load_state_dict(torch.load(config.model_path_generator))
        # model_decode = torch.load(config.model_path_whole_decoder).to(config.device)
        # model_generator = torch.load(config.model_path_whole_generator).to(config.device)
        model_decoder.eval()
        model_generator.eval()

    torch.cuda.empty_cache()
    while True:
        # 接收数据
        dat, src_mask = q_enc.get()
        # gpu_tracker = MemTracker() 
        torch.cuda.empty_cache()
        src_enc = dat.clone()
        del dat
        torch.cuda.empty_cache()
        # gpu_tracker.track()                  # 开始检测    
        translation = translate_decode_split(src_enc, src_mask, model_decoder, model_generator, use_beam=True)
        # gpu_tracker.track()                  # 开始检测    
        q_result.put(translation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model PipeLine N Process Predict')
    parser.add_argument('--datafile', type=str, required=1, default="", help='data file name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--instances', type=int, default=1, help='total pipelines')
    parser.add_argument('--show_result', type=int, default=0, help='show result')
    parser.add_argument('--logfile', type=str, default="report.txt", help='log file name')
    args = parser.parse_args()
    datafile = args.datafile
    logfile = args.logfile
    batch_size = args.batch_size
    show_result = args.show_result
    total_pipelines = args.instances

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    mp.set_start_method('spawn')

    # 加载数据文件
    txts = readtxt(datafile)
    if txts:
        sentences = txts.splitlines()
        sentences = list(filter(None, sentences))
        total_sent = len(sentences)
    else:
        print('data file error')
        sys.exit()

    '''
    # GPU记录
    obj = GPU_MEM()
    obj.build()
    obj.start()
    '''

    # 开始计时
    start = time.time()
    print(' NMT Task: Multi Pipelines '.center(40, '-'))
    print('total pipelines:%d' % total_pipelines)
    print('total sentences:%d' % total_sent)

    print('Building model...')
    # gpu_tracker.track()                  # 开始检测
    
    # 创建拆分后的模型
    model_encoder, model_decoder, model_generator = make_split_model(
                        config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_encoder.share_memory()
    model_decoder.share_memory()
    model_generator.share_memory()
   
    # 加载模型 
    print('Loading model...')
    model_encoder.load_state_dict(torch.load(config.model_path_encoder))
    model_decoder.load_state_dict(torch.load(config.model_path_decoder))
    model_generator.load_state_dict(torch.load(config.model_path_generator))

    model_encoder.eval()
    model_decoder.eval()
    model_generator.eval()
    torch.cuda.empty_cache()
    # gpu_tracker.track()                  # 开始检测
    # 加载模型结束
    
    '''
    # ----- 以下为另一种模型加载方式 -----
    # 加载模型
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
    model_generator = model.generator

    model_encoder = Transformer_encode(encoder, src_embed)
    model_decoder = Transformer_decode(decoder, tgt_embed)
    torch.cuda.empty_cache()
    # 加载模型结束
    '''
    # gpu_tracker.track()                  # 开始检测
    print('create Queue...')
    # total_pipelines = 5
    # 创建时间统计队列和流水线队列 
    tmr_queue = mp.Queue()   
    queue_list = []
    for i in range(total_pipelines):
        # 流水线队列
        q_text = mp.Queue()
        q_enc = mp.Queue()
        q_result= mp.Queue()
        queue_list.append ([q_text, q_enc, q_result])

    print('create process...')
    sub_process = []
    for i in range(total_pipelines):
        # 流水线进程
        p_encode = mp.Process(target=proc_encode, args=(queue_list[i][0], queue_list[i][1], model_encoder))
        p_decode = mp.Process(target=proc_decode, args=(queue_list[i][1], queue_list[i][2], model_decoder, model_generator))
        sub_process.append ([p_encode, p_decode])

    print('process start...')
    for i in range(total_pipelines):
        sub_process[i][0].start()
    print('decoder start....')
    for i in range(total_pipelines):
        sub_process[i][1].start()
    # 计时为加载模型时间
    loadstime = (time.time() - start) * 1000

    print('Create sender process...')
    sender_list = []
    for i in range(total_pipelines):
       p_sender = mp.Process(target=send_dat, args=(sentences, queue_list[i][0], queue_list[i][2], batch_size, show_result, tmr_queue))
       sender_list.append(p_sender)

    # 开始计时
    start = time.time()
    for i in range(total_pipelines):
        sender_list[i].start()

    print('process join...')
    for i in range(total_pipelines):
        sender_list[i].join()
    
    # gpu_tracker.track()                  # 开始检测
    mem_data = []
    mem_ave, mem_max = 0,0

    # 显存记录器停止并输出结果
    '''
    obj.stop()
    mem_data = obj.data
    mem_ave = obj.mem_ave()
    mem_max = obj.mem_max()
    '''
    print('mem_ave:', mem_ave)
    print('mem_max:', mem_max)

    memory = GPU_memory(0)
    predict_time = (time.time() - start) * 1000
    avetime = predict_time / total_sent
    # 从计时队列中获取每个pipeline的运行时间
    pipe_times = []
    for i in range(total_pipelines):
        tmr = tmr_queue.get()
        pipe_times.append(tmr)
    ave_pipe_time = np.average (pipe_times)

    print('加载模型用时:%.3f 毫秒' % loadstime)
    print('Used Memory:%d MB' % memory)
    print('预测总计用时:%.3f 毫秒' % predict_time)
    print('预测单句用时:%.3f 毫秒' % avetime)
    print('Pipeline用时(毫秒):%s ' % pipe_times)
    print('Pipeline平均用时:%.3f 毫秒' % ave_pipe_time)
    print('平均单句用时:%.3f 毫秒' % (ave_pipe_time / total_sent))
    print('任务总计用时:%.3f 毫秒' % (predict_time + loadstime) )

    result = {'name':'Multi Pipeline', 'total_pipelines': total_pipelines, 
                'total_sent':total_sent,'memory':memory, 
                'loadstime': loadstime, 'predict_time':predict_time, 
                'avetime':avetime, 'batch_size':batch_size,
                'pipe_times':pipe_times, 'ave_pipe_time':ave_pipe_time,                
                'mem data':mem_data, 'mem_ave':mem_ave, 'mem_max':mem_max
             }

    # 追加到日志文件
    savetofile(json.dumps(result), logfile)

    for i in range(total_pipelines):
        sub_process[i][0].terminate()
        sub_process[i][1].terminate()
    
