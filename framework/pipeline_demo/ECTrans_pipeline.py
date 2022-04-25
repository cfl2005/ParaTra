#!/usr/bin/env python3
#coding:utf-8

import argparse
# from multiprocessing.queues import Queue
import os
import sys
import time
import zmq
import random
# import re
import json
import logging
import numpy as np
from tqdm import tqdm
import pprint

import torch
#from torch.utils.data import DataLoader
#from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

import utils
import config
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

gblgebug = True

def debug(func):
    global gbldebug 
    def wrapTheFunction(gbldebug):
        if gbldebug:
           func()
     
    return wrapTheFunction

class SmratRouter():
    def __init__(self, sender, n=2) -> None:
        # self.queuelist = queuelist
        self.total_queue = n 
        self.queue_index = 0
        self.sender = sender

    def send_json(self, dat):
        self.sender.send_json(dat)
        self.queue_index = (self.queue_index + 1 ) % self.total_queue
        
def server_req(ip, port, port_task, workers): #*queuelist
    # @debug
    print("server_req: %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://%s:%d"%(ip, port_task))

    smart_router = SmratRouter(sender, n=workers)

    while True:
        ret = receiver.recv_json()
        smart_router.send_json(ret)
        # sender.send_json(ret)
        receiver.send('OK'.encode())

def proc_encode(ip, port_task, q_enc, model_encoder):
    consumer_id = random.randrange(1000,9999)
    print("proc_encode ID: #%s %d ==> Queue" % (consumer_id, port_task) )
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port_task))

    while True:
        # id, dat = q_text.get()
        data = consumer_receiver.recv_json()
        wid = data['id']
        dat = data['texts']        
        # 转为Tensor            
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        # print('src_enc:',type(src_enc))
        # print('encoder:%s' % consumer_id)
        q_enc.put( (wid, src_enc, src_mask) )
        torch.cuda.empty_cache()

def proc_decode(q_enc, ip, port_out_pub, model_decoder, model_generator):
    # send work
    context = zmq.Context()
    # zmq_socket = context.socket(zmq.REQ)
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect("tcp://%s:%d"%(ip, port_out_pub))

    while True:
        id, dat, src_mask = q_enc.get()
        torch.cuda.empty_cache()
        src_enc = dat.clone()
        del dat
        torch.cuda.empty_cache()
        translation = translate_decode_split(src_enc, src_mask, model_decoder, model_generator, use_beam=True)
        torch.cuda.empty_cache()
        result = {'id':id, 'result':translation}
        # print('result:', result)
        zmq_socket.send_json(result)
        # message = zmq_socket.recv()

def result_pub(ip, port_out_pub, port_out):
    print("result publisher: %d ==> %d " % (port_out_pub, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port_out_pub))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        publisher.send_json(ret)
        # print('publish dat')
        # title = str(ret)[:25]
        # print('publish:%s' % title)

def result_collector(ip, port_out, total, task_id, result_queue):
    print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))

    filter_title = "{\"id\":\"%s" % task_id
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
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5559,
                port_out=5560,
                workers=2):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        # self.port_out_encoder = port + 200 
        # self.port_in_decoder = port + 300 
        self.port_out_pub = port_out + 100
        self.port_out = port_out
        self.workers = workers
        self.queues = []

        self.p_workers = []
        self.server_work = None
        self.publisher = None

    def start(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(1)
        #　['fork', 'spawn', 'forkserver']
        mp.set_start_method('spawn')

        #　{'file_system', 'file_descriptor'}
        # mp.set_sharing_strategy('file_system')
        '''
        strategies = mp.get_all_sharing_strategies()
        print('get_all_sharing_strategies:', strategies )
        print('strategies:', mp.get_sharing_strategy() )
        methods = mp.get_all_start_methods()
        print('get_all_start_methods:', methods )
        print('method :', mp.get_start_method() )
        '''

        for i in range(self.workers):
            # q_text = mp.Queue()
            q_encoder = mp.Queue()
            # self.queues.append([q_text, q_encoder])
            self.queues.append(q_encoder)
        
        model_encoder, model_decoder, model_generator = make_split_model(
                            config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)
        model_encoder.share_memory()
        model_decoder.share_memory()
        model_generator.share_memory()
       
        print('Loading model...')
        model_encoder.load_state_dict(torch.load(config.model_path_encoder))
        model_decoder.load_state_dict(torch.load(config.model_path_decoder))
        model_generator.load_state_dict(torch.load(config.model_path_generator))

        model_encoder.eval()
        model_decoder.eval()
        model_generator.eval()
        torch.cuda.empty_cache()

        self.p_workers = []
        for i in range(self.workers):
            p_encode = mp.Process(target=proc_encode, 
                                    args=(self.ip, self.port_task,
                                        self.queues[i], model_encoder))
                                    #args=(self.queues[i][0], self.queues[i][1], model_encoder))

            p_decode = mp.Process(target=proc_decode, 
                                    args=(self.queues[i], 
                                    self.ip, self.port_out_pub,
                                    model_decoder, model_generator))

            self.p_workers.append ([p_encode, p_decode])

        print('encoder start...')
        for i in range(self.workers):
            self.p_workers[i][0].start()

        print('decoder start....')
        for i in range(self.workers):
            self.p_workers[i][1].start()

        print('publisher start....')
        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_pub, self.port_out))
        self.publisher.start()

        print('server_work start....')
        # qqs = [x[0] for x in self.queues]
        self.server_work = mp.Process(target=server_req, 
                                args=(self.ip, self.port, self.port_task, self.workers)) #self.queues
        self.server_work.start()

        print('ECTrans server ready....')

    def stop(self):
        # self.trans_work.terminate()
        # self.trans_work.join()
        for i in range(self.workers):
            self.p_workers[i][0].terminate()
            self.p_workers[i][1].terminate()
            self.p_workers[i][0].join()
            self.p_workers[i][1].join()

    def __enter__(self):
        pass

    def __exit__(self):
        pass
        self.stop()

    def join(self):
        pass

#-----------------------------------------
class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5544,
                port_out=5545,
                batch_size=8):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = batch_size
        self.result = []
        self.total = 0
        self.collector = None

    def send(self, datafile):
        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        batch_size = self.batch_size
        total = len(sentences)
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)
        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        print('task_id:', task_id)
        print('total sample:', total)
        print('total_batch:', total_batch)
        self.total = total

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))
        
        start = time.time()
        print('start...')

        for i in tqdm(range(total_batch), ncols=80):
            txts = sentlist[i]
            batch_text = get_sample(txts).numpy().tolist()
            wid = '%d_%d' % (task_id, i)
            work_message = {'id':wid, 'texts': batch_text}
            zmq_socket.send_json(work_message)
            message = zmq_socket.recv()

        print('wait data return...')
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
    parser = argparse.ArgumentParser(description='ECTrans')
    parser.add_argument('--cmd', type=str, required=True, default="", help=' server, client')
    parser.add_argument('--workers', type=int, default=2, help='pipeline number')
    parser.add_argument('--datafile', type=str, default="data_100.txt", help='file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    args = parser.parse_args()

    cmd = args.cmd
    datafile = args.datafile
    workers = args.workers
    batch_size = args.batch_size

    if cmd=='server':
        # python3 ECTrans.py --cmd=server --workers=2
        print('start server...')
        server = ECTrans_Server(ip='127.0.0.1',
                                port=5557,
                                port_out=5560,
                                workers=workers)
        server.start()

    if cmd=='client':
        print('start...')
        client = ECTrans_Client(ip='127.0.0.1',
                            port=5557,
                            port_out=5560,
                            batch_size=batch_size)
        print('send data...')                            
        sents = client.send(datafile)

        print('total results :%d' % len(sents))

