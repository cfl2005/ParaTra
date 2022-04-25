#!/usr/bin/env python3
#coding:utf-8

'''
ZMQ FrameWork + Task
'''

import argparse
import os
import sys
import time
import zmq
import random
import re
import json
import logging
import numpy as np
from tqdm import tqdm
import pprint

import torch
import torch.multiprocessing as mp

import utils
import config
from train import translate
from model import make_model

from translate import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')


def server_req(ip, port, port_task):
    print("server_req: %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://%s:%d"%(ip, port_task))

    while True:
        ret = receiver.recv_json()
        sender.send_json(ret)
        receiver.send('OK'.encode())
        cid = ret.get('id', '')
        length = ret.get('length', '')
        etime = time.time()*1000
        print('Task id:%s, enter_time:%s, length:%4d'% (cid, etime, length))

def consumer(ip, port_task, port_out_publisher, batch_size):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.share_memory()
    print('Loading model...')
    #model.load_state_dict(torch.load(os.path.join('../../', config.model_path)))
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    torch.cuda.empty_cache()

    # Task ID
    consumer_id = random.randrange(1000,9999)
    print("consumer ID: #%s" % (consumer_id) )

    #create ZMQ
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port_task))
    
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out_publisher))
    
    while True:
        data = consumer_receiver.recv_json()
        etime = time.time()
        wid = data['id']
        sentences = data['texts']
        embedding = data['embedding']
        length = data['length']
        print('consumer ID: #%s => cid:%s total:%d start_time:%s' % (consumer_id, wid, length,etime))
        result = translate_batch(sentences, model, 
                                batch_size=batch_size,
                                embedding=embedding)
        
        jsdat = {'id': wid, 'result': result, 'consumer':consumer_id}
        consumer_sender.send_json(jsdat)
        ftime = time.time()
        print('consumer ID: #%s =>cid: %s finished time: %s' % (consumer_id, wid, ftime))

def result_pub(ip, port, port_out):
    print("result publisher: %d ==> %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        publisher.send_json(ret)
        title = str(ret)[:25]
        # print('publish:%s' % title)

def result_collector(ip, port_out, total, task_id, result_queue):
    # print("result_collector:  ==> %d " % (port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://%s:%d"%(ip, port_out))
    # print('result_collector:%d => cid:%s'% (port_out, task_id))

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
        wid = ret['id']
        total_result += t_sents
        
        if total_result >= total:break

#-----------------------------------------
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5588,
                port_out=5560,
                workers=4, 
                batch_size=8):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        self.port_out_publisher = port_out + 100
        self.port_out = port_out
        self.workers = workers
        self.batch_size = batch_size

    def start(self):
        # crate
        p_workers = []
        for i in range(self.workers):
            p = mp.Process(target=consumer, 
                            args=(self.ip, self.port_task, 
                                    self.port_out_publisher,
                                    self.batch_size))

            p_workers.append(p)

        # start
        for i in range(self.workers):
            p_workers[i].start()

        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_publisher, self.port_out))
        self.publisher.start()

        self.server = mp.Process(target=server_req, 
                                args=(self.ip, self.port, self.port_task))
        self.server.start()
                
        print('ECTrans Task server ready....')

    def stop(self):
        pass
        for i in range(self.workers):
            p_workers[i].terminate()

    def __enter__(self):
        pass

    def __exit__(self):
        pass
        for i in range(self.workers):
            p_workers[i].terminate()

#-----------------------------------------
class ECTrans_Client():
    '''client
    '''

    def __init__(self,
                ip='127.0.0.1',
                port=5588,
                port_out=5560,
                batch_size=8):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = batch_size
        self.result = []
        self.total = 0
        self.collector = None

    def send_batch(self, datafile, embedding=0):
        txts = readtxt(datafile)

        sentences = list(filter(None, txts.splitlines()))
        batch_size = self.batch_size
        total = len(sentences)
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)

        task_id = str(int(time.time()*1000)*1000 + random.randrange(1000, 9999))
        self.total = total

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))

        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))

        start = time.time()
        print('start :%f' % start)
        embed_total = 0
        #for i in tqdm(range(total_batch), ncols=80):
        for i in range(total_batch):
            # wid = '%s_%d' % (task_id, i)
            txts = sentlist[i]
            len_dat = len(txts)
            embed_start = time.time()
            if embedding==1:
                batch_text = get_sample(txts).numpy().tolist()
            else:
                batch_text = txts
            embed_time = (time.time() - embed_start)*1000
            embed_total += embed_time
            # print('embedding_time:%f' % embed_time)
            
            work_message = {'id': task_id, 'texts':batch_text, 'embedding':embedding, 'length':len_dat}

            zmq_socket.send_json(work_message)
            message = zmq_socket.recv()

        print('embedding time:%f' % embed_total )
        #print('all data send, wait for response...')
        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
        
        self.collector.terminate()
        self.collector.join()

        #print('result:', len(result))
        end_time = time.time()
        print('task end time:%f' % end_time )
        #predict_time = (time.time() - start)*1000
        #avetime = predict_time/total
        return result 
    
    def send(self, datafile, embedding=0):
        txts = readtxt(datafile)

        sentences = list(filter(None, txts.splitlines()))
        total = len(sentences)
        self.total = total

        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        task_id = str(task_id)
        btime = int(time.time()*1000)
        print('building task:%s, task_id:%s, length:%d' % (btime, task_id, total))

        self.task_id = task_id

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))

        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))

        start = time.time()
        if embedding==1:
            batch_text = get_sample(sentences).numpy().tolist()
        else:
            batch_text = sentences

        work_message = {'id': task_id, 'texts':batch_text, 'embedding':embedding, 'length':total}
        zmq_socket.send_json(work_message)
        message = zmq_socket.recv()
        # print('all data send, wait for response...')
        embedding_time = (time.time() - start)*1000
        print('embedding time:%f ms' % embedding_time )

        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            # print('result received...')
            if len(result) >= total:break;
        
        self.collector.terminate()
        self.collector.join()

        #print('result:', len(result))

        predict_time = (time.time() - start)*1000
        gpu_time = (predict_time - embedding_time)
        print('gpu time:%f 毫秒' % gpu_time )
        print('total time :%f 毫秒' % predict_time )

        #avetime = predict_time/total
        #print('预测单句用时:%f 毫秒' % avetime )
        return result 
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--cmd', type=str, default="server", help='server / client')
    parser.add_argument('--datafile', type=str, default='./report/data_100.txt', help='datafile')
    parser.add_argument('--workers', type=int, default=4, help='workers, default:4')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size, default:8')
    parser.add_argument('--embedding', type=int, default=0, help='embedding, default:0')
    parser.add_argument('--batch', type=int, default=0, help='batch, default:0')
    parser.add_argument('--debug', type=int, default=1, help='debug, default:0')
    args = parser.parse_args()

    cmd = args.cmd
    workers = args.workers
    batch_size = args.batch_size
    datafile = args.datafile
    embedding = args.embedding
    debug = args.debug
    batch = args.batch
   
    if cmd=='server':
        print('start server...')
        server = ECTrans_Server(
                        ip='0.0.0.0',
                        port=5588,
                        port_out=5560,
                        workers=workers, 
                        batch_size=batch_size)
        server.start()

    if cmd=='client':
        print('strat client...')
        client = ECTrans_Client(ip='127.0.0.1',
                            port=5588,
                            port_out=5560,
                            batch_size=batch_size)
        # 开始计时
        start = time.time()

        print('send data...')
        if batch:
            rst = client.send_batch(datafile, embedding=embedding)
        else:
            rst = client.send(datafile, embedding=embedding)
        # print('result:\n%s' % '\n'.join(rst) )

        total = len(rst)
        print('total results :%d' % total)
        predict_time = (time.time() - start)*1000
        avetime = predict_time/total
        print('total time:%f 毫秒' % predict_time )
        print('single time:%f 毫秒' % avetime)


