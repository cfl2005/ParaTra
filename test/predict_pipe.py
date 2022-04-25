#!/usr/bin/env python3
#coding:utf-8

import argparse
import os
import re
import json
import logging
#from flask import Flask, request, render_template, jsonify, json=
#import multiprocessing
import time

import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from train import translate, translate_encode, translate_decode
from utils import english_tokenizer_load
from model import make_model
import torch.multiprocessing as mp
#from torchvision import datasets, transforms
from torch.multiprocessing import Pool, Manager, Queue

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')

def get_sample(sent):
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    #batch_input = torch.LongTensor(np.array(src_tokens)).to(torch.device('cpu')) #config.device)
    return src_tokens

def translate_sample(txt, model, beam_search=True):
    print('text:', txt)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
    batch_input = torch.LongTensor(np.array(sample)).to(config.device)
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)


def translate_sample1(txts, model, beam_search=True):
    res = {}
    print('texts:', txts)
    batch_input = []

    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3

    for txt in txts.splitlines():
        sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
        batch_input.append(torch.LongTensor(np.array(sample)).to(config.device))

    #batch_input = get_sample(txt)
    #batch_input = 
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)

def send_dat(texts, q_text, q_result):
    print('sending text...')
    dt = texts.splitlines()
    total = len(dt)
    for i,txt in enumerate(dt):
        batch_input = get_sample(txt)
        print('batch_input:',type(batch_input))
        print("send dat: %s" %i)
        q_text.put(batch_input)
    
    i = 0 
    print('receive result...')
    while 1:
        ret = q_result.get()
        if type(ret)==str:
            print('result:', ret)
        else:
            print('result:', type(ret))
        i+=1
        if i>=total:
            break;
    print('work end...')
    

def proc_encode(q_text, q_enc, model):
    while True:
        print("proc_encode.")
        dat = q_text.get()
        # print('q_text dat:',type(dat))
        batch_input = torch.LongTensor(np.array(dat)).to(config.device)
        src_enc, src_mask = translate_encode(batch_input, model)
        # print('src_enc:',type(src_enc))
        q_enc.put( (src_enc,src_mask) )

def proc_decode(q_enc, q_result, model):
    while True:
        print("proc_decode rev:")
        dat, src_mask = q_enc.get()
        #if not dat is None:
        src_enc = dat.clone()
        del dat
        #print('src_enc:',type(src_enc))
        translation = translate_decode(src_enc, src_mask, model, use_beam=True)
        #print('translation:', translation)
        q_result.put(translation)

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=1, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')


    args = parser.parse_args()

    start = time.time()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')
    #mp.set_start_method('forkserver')

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    model.share_memory()
    print('Loading model...')
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
  
    print('create Queue...')
    q_text = Queue()
    q_enc = Queue()
    q_result = Queue()

    processes = []
    texts = '''Moreover, this war has been funded differently from any other war in America’s history –perhaps in any country’s recent history.
    But China’s policy failures should come as no surprise.
    The results have been devastating.
    Wind and solar are often presented as alternatives to oil and gas, but they cannot compete with traditional sources for electricity generation.
    Nor are Muslim women alone.'''
    
    print('create process...')
    p_encode = mp.Process(target=proc_encode, args=(q_text, q_enc, model))
    p_decode = mp.Process(target=proc_decode, args=(q_enc, q_result, model))

    print('process start...')
    p_decode.start()
    p_encode.start()

    estime = (time.time() - start)*1000

    start = time.time()

    print('Create sender process...')
    p_sender = mp.Process(target=send_dat, args=(texts, q_text, q_result))
    p_sender.start()

    print('process join...')
    #p_encode.join()
    #p_decode.join()
    p_sender.join()
    
    #send_dat(texts, q_text)

    estime = (time.time() - start)*1000
    print('use time:%f ms' % estime )

