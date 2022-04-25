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
from sc_lib import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

def server_req_queue(ip, port, queue_hrrn):
    '''server
    '''
    print("server_req: %d ==> task_dat " % (port))
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    while True:
        dat = receiver.recv_json()
        cid = dat.get('id')
        length = dat.get('length')
        queue_hrrn.put(dat)

        receiver.send('OK'.encode())
        print('Response to Cleitn: task id:%s ==>OK  length:%d'% (cid, length))


def task_hrrn(queue_hrrn, queue_task):
    print("task_hrrn: task_data ==> queue_task ") # % (port_task) )
    task_data = {}
    
    objhrrn = HRRN()

    while True:
        time.sleep(0.01)
        if queue_hrrn.qsize() > 0:
            #data = queue_hrrn.get_nowait()
            task = queue_hrrn.get()
            tid = task.get('id', None)
            # task_data.append(task)
            task_data.update({tid: task})
            print('task_data length:', len(task_data))
            
            dat_length = task.get('length')
            objhrrn.add(dat_length, tid=int(tid))
            # 排序
            pll()
            print('data:\n', objhrrn.text())
            pl()
            print('sort...')
            objhrrn.hrrn()
            pll()

        que_count = queue_task.qsize()
        task_count = len(task_data)
        '''
        print('task_data:', type(task_data))
        print('task_count:%d'% task_count)
        print('que_count:%d'% que_count)
        '''

        if task_count > 0 and que_count < 1:
            dat = None
            #print(objhrrn.text())
            #objhrrn.hrrn()
            print('task_ids:', objhrrn.task_ids)
            ids = objhrrn.pop()
            print('pop ids:', ids)
            dat = task_data.pop(str(ids))
            
            if not dat is None:
                tid = dat.get('id', None)
                # 发送任务
                queue_task.put(dat)
                print('send task:%s'% tid)

def consumer(ip, task_queue, port_out_publisher, batch_size): #port_task

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

    context = zmq.Context()
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://%s:%s"%(ip, port_out_publisher))
    
    while True:
        data = task_queue.get()
        wid = data['id']
        sentences = data['texts']
        embedding = data['embedding']
        stime = time.time()
        length = data['length']

        print('consumer ID: #%s => cid:%s total:%d start_time:%s' % (consumer_id, wid, length, stime))
        result = translate_batch(sentences, model, 
                                batch_size=batch_size,
                                embedding=embedding)
        
        jsdat = {'id': wid, 'result': result, 'consumer':consumer_id}
        consumer_sender.send_json(jsdat)
        print('consumer ID: #%s =>cid: %s finished' % (consumer_id, wid))

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
        #cid = ret.get('id', 'noid')
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
                port=6588,
                port_out=6500,
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
        self.task_data = mp.Manager().list()
        #manager = mp.multiprocessing.Manager()
        #self.task_data = manager.list()

        self.queue_task = mp.Queue()
        self.queue_hrrn = mp.Queue()
        
        print('正在启动:工作进程...')
        p_workers = []
        for i in range(self.workers):
            p = mp.Process(target=consumer, 
                            args=(self.ip, self.queue_task,  #self.port_task
                                    self.port_out_publisher,
                                    self.batch_size))

            p_workers.append(p)

        for i in range(self.workers):
            p_workers[i].start()
        
        print('正在启动:task_svr...')
        self.task_svr = mp.Process(target=task_hrrn,
                                   args=(self.queue_hrrn, self.queue_task))
        self.task_svr.start()

        self.publisher = mp.Process(target=result_pub,
                                    args=(self.ip, self.port_out_publisher, self.port_out))
        self.publisher.start()

        self.server = mp.Process(target=server_req_queue, 
                                args=(self.ip, self.port, self.queue_hrrn))
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
    def __init__(self,
                ip='127.0.0.1',
                port=6588,
                port_out=6500,
                batch_size=8):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = batch_size
        self.result = []
        self.total = 0
        self.collector = None
        self.task_id = ''

    def send_batch(self, datafile, embedding=0):
        txts = readtxt(datafile)

        sentences = list(filter(None, txts.splitlines()))
        batch_size = self.batch_size
        total = len(sentences)
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]

        total_batch = len(sentlist)

        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        # print('total:', total)
        # print('total_batch:', total_batch)
        self.total = total
        self.task_id = str(task_id)

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))

        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))

        for i in range(total_batch):
            wid = '%d_%d' % (task_id, i)
            txts = sentlist[i]
            if embedding==1:
                batch_text = get_sample(txts).numpy().tolist()
            else:
                batch_text = txts

            work_message = {'id': wid, 'texts':batch_text, 'embedding':embedding}
            zmq_socket.send_json(work_message)
            message = zmq_socket.recv()
        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            if len(result) >= total:break;
        
        self.collector.terminate()
        self.collector.join()

        return result 
    
    def send(self, datafile, embedding=0):
        txts = readtxt(datafile)

        sentences = list(filter(None, txts.splitlines()))
        total = len(sentences)
        self.total = total

        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        task_id = str(task_id)
        btime = int(time.time()*1000)
        print('enter time:%s, task_id:%s, length:%d' % (btime, task_id, total))
        self.task_id = task_id

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))

        self.collector.start()

        context = zmq.Context()
        zmq_socket = context.socket(zmq.REQ)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))

        if embedding==1:
            batch_text = get_sample(sentences).numpy().tolist()
        else:
            batch_text = sentences

        work_message = {'id': task_id, 'texts':batch_text, 'embedding':embedding, 'length': total}
        zmq_socket.send_json(work_message)
        message = zmq_socket.recv()
        # print('all data send, wait for response...')


        result = []
        while 1:
            ret = result_queue.get()
            result.extend(ret)
            # print('result received...')
            if len(result) >= total:break;
        
        self.collector.terminate()
        self.collector.join()

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
    parser.add_argument('--debug', type=int, default=1, help='debug, default:0')
    args = parser.parse_args()

    cmd = args.cmd
    workers = args.workers
    batch_size = args.batch_size
    datafile = args.datafile
    embedding = args.embedding
    debug = args.debug
   
    if cmd=='server':
        print('server start...')
        server = ECTrans_Server(
                        ip='0.0.0.0',
                        port=6588,
                        port_out=6500,
                        workers=workers, 
                        batch_size=batch_size)
        server.start()

    if cmd=='client':
        print('client start...')

        client = ECTrans_Client(ip='127.0.0.1',
                            port=6688,
                            port_out=6560,
                            batch_size=batch_size)
        start = time.time()
        print('send data...')
        rst = client.send(datafile, embedding=embedding)
        total = len(rst)
        print('total results :%d' % total)