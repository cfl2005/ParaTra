#!/usr/bin/env python3
#coding:utf-8

import argparse
import os
import sys
import time
import zmq
import random
import json
import logging
import numpy as np
from tqdm import tqdm
import pprint

import torch
import torch.multiprocessing as mp
import ECTrans_config

import warnings
warnings.filterwarnings('ignore')

gblgebug = True

def debug(func):
    global gbldebug 
    def wrapTheFunction(gbldebug):
        if gbldebug:
           func()
     
    return wrapTheFunction

def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''

def savetofile(txt, filename, encoding='utf-8', method='a+'):
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt)+ '\n')
        return 1
    except :
        return 0

def result_collector(ip, port_out, total, task_id, result_queue):
    print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    filter_title = "{\"client_id\":\"%s" % task_id
    receiver.setsockopt(zmq.SUBSCRIBE, filter_title.encode())
    # print('filter_title:', filter_title)
    
    collecter_data = {}
    total_result = 0
    while True:
        ret = receiver.recv_json()
        sents = ret['result']
        result_queue.put(sents)

        t_sents = len(sents)
        total_result += t_sents

        if total_result >= total: break

    if collecter_data: pprint.pprint(collecter_data)
#-----------------------------------------
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5544,
                port_out=5545):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = ECTrans_config.batch_size
        self.result = []
        self.total = 0
        self.collector = None
        self.encoder = None
    
    def set_encoder(self, fun):
        self.encoder = fun

    def send(self, datafile):
        if self.encoder is None:
            raise ValueError("set_encode")

        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        
        total = len(sentences)

        self.total = len(sentences)
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        
        start = time.time()
        print('start...')

        txts = sentences
        batch_text = self.encoder(txts)
        work_message = {'tid':task_id, 'texts': batch_text, 'length': total}
        zmq_socket.send_json(work_message)
        message = zmq_socket.recv()

        print('wait return...')
        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
        
        self.collector.terminate()
        self.collector.join()

        predict_time = (time.time() - start)*1000
        avetime = predict_time/total
        return result 
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    sys.path.append("../../../")
    from translate import get_sample

    parser = argparse.ArgumentParser(description='ECTrans')
    parser.add_argument('--datafile', type=str, default="data_100.txt", help='file')
    args = parser.parse_args()
    datafile = args.datafile

    print('start client...')

    client = ECTrans_Client(ip='127.0.0.1',
                        port=5557,
                        port_out=5560,)

    txt_encode = lambda x: get_sample(x).numpy().tolist()
    client.set_encoder(txt_encode)

    sents = client.send(datafile)

    print('total results :%d' % len(sents))

