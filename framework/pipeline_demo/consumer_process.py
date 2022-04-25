#!/usr/bin/env python3
#coding:utf-8

import os
import sys
import time
import zmq
import random

import argparse
import os
#sys.path.append('../../')

import re
import json
import logging
import sys
import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

import utils
import config
from train import translate
from model import make_model

from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

def consumer(ip, port, port_out):
    '''
    import random
    import time
    import zmq
    '''

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    #model.load_state_dict(torch.load(os.path.join('../../', config.model_path)))
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    torch.cuda.empty_cache()

    consumer_id = random.randrange(1000,9999)
    print("consumer ID: #%s" % (consumer_id) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out))
    
    while True:
        data = consumer_receiver.recv_json()
        wid = data['id']
        sentences = data['texts']

        batch_size = 8
        result = translate_batch(sentences, model, batch_size=batch_size)
        
        jsdat = {'id': wid, 'result':result, 'consumer':consumer_id,}
        consumer_sender.send_json(jsdat)


if __name__ == '__main__':

    workers = 2
    if len(sys.argv)==2:
        workers = int(sys.argv[1])
 
    p_workers = []
    for i in range(workers):
        p = mp.Process(target=consumer, args=('127.0.0.1', 5557, 5558) )
        p_workers.append(p)

    for i in range(workers):
        p_workers[i].start()
