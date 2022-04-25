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
from collections import OrderedDict

import utils
import config
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

import ECTrans_config
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

def que_process(ip, port_task, cache):
    print("que_process: connect PUSH %d==> Queue " % (port_task) )
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://%s:%d"%(ip, port_task))
    
    while 1:
        dat_msg = None
        cur_length = 0
        while 1:
            if cache.qsize() > 0:
                dat = cache.get()
                if cur_length ==0:
                    dat_msg = dat
                    cur_length = dat['dat_len']
                else:
                    tid, (b,e) = dat['client_ids'][0]
                    texts = dat['texts']
                    dat_len = dat['dat_len']
                    ids = (tid, (b + cur_length, e + cur_length))
                    dat_msg['client_ids'].append(ids)
                    dat_msg['texts'].extend(texts)
                    dat_msg['dat_len'] = cur_length 
                    cur_length += dat_len
                
            if cache.qsize() ==0 or cur_length >= ECTrans_config.batch_size : break
        
        if dat_msg:
            sender.send_json(dat_msg)
        time.sleep(ECTrans_config.time_windows/1000)

def real_pub(ip, port, port_out):
    print("publisher: PULL %d ==> PUSH %d " % (port, port_out) )
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port))

    publisher = context.socket(zmq.PUSH)
    publisher.bind("tcp://%s:%d"%(ip, port_out))

    while True:
        ret = receiver.recv_json()
        ids = ret.get('client_ids', 'null')
        publisher.send_json(ret)

class SmratRouter():
    def __init__(self, ip, port_task, batch_size, batch_value): # cache,  -> None
        self.queue_index = 0
        #self.senders = senders
        self.cache = mp.Queue()         #mp.Queue()   cache
        self.timestrap = time.time()
        self.cache_length = 0
        self.process = None
        self.ip = ip
        self.port_task = port_task

        self.batch_value = batch_value
        self.batch_size = batch_size

        context = zmq.Context()
        self.sender_0 = context.socket(zmq.PUSH)
        self.sender_0.connect("tcp://%s:%d"%(ip, port_task+2))
        self.sender_1 = context.socket(zmq.PUSH)
        #self.sender_1.bind("tcp://%s:%d"%(ip, port_task+1))
        self.sender_1.connect("tcp://%s:%d"%(ip, port_task+1))
        self.senders = [self.sender_0, self.sender_1]


    def send_json(self, dat):
        isbatch = 0
        length = len(dat['texts'])
        print('texts length:', length)
        if length >= self.batch_value:
            isbatch = 1
        else:
            isbatch = 0
        # print("isbatch:", isbatch)
        tid = dat['tid']
        sentences = dat['texts']

        total = length
        batch_size = self.batch_size
        sentlist = [sentences[i*batch_size:(i+1)*batch_size] 
                    for i in range(np.ceil(total/batch_size).astype(int))]
        total_batch = len(sentlist)
        # print('total_batch:', total_batch)

        for i in range(total_batch):
            batch_text = sentlist[i]
            blen = len(batch_text)
            wid = '%d_%d' % (tid, i)
            ids = [(wid, (0, blen))]
            # print('ids:', ids, 'blen:', blen)
            # print(blen == self.batch_size)
            dat_msg = {'client_ids':ids, 'texts': batch_text, 'dat_len': blen}
            if blen == self.batch_size:
                print('send to port')
                self.senders[isbatch].send_json(dat_msg)
            else:
                print('send to queue')
                self.cache.put(dat_msg)
                  
    def start(self):
        pass 
        self.process = mp.Process(target=que_process, 
                                args=(self.ip, self.port_task+2, self.cache))
        self.process.start()

def server_route(ip, port, port_back):
    print("server_route: ROUTER %d ==> %d " % (port, port_back) )
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind("tcp://%s:%d"%(ip, port))
    
    backend = context.socket(zmq.ROUTER)
    backend.bind("tcp://%s:%d"%(ip, port_back))

    frontend.setsockopt(zmq.RCVHWM, 100)
    backend.setsockopt(zmq.RCVHWM, 100)
     
    workers = OrderedDict()
    clients = {}
    msg_cache = []
    poll = zmq.Poller()

    poll.register(backend, zmq.POLLIN)
    poll.register(frontend, zmq.POLLIN)

    while True:
        socks = dict(poll.poll(10))
        now = time.time()
        if backend in socks and socks[backend] == zmq.POLLIN:
            worker_addr, client_addr, response = backend.recv_multipart()
            workers[worker_addr] = time.time()
            if client_addr in clients:
                frontend.send_multipart([client_addr, response])
                clients.pop(client_addr)
            else:
                pass
        while len(msg_cache) > 0 and len(workers) > 0:
            worker_addr, t = workers.popitem()
            if t - now > 1:
                continue
            msg = msg_cache.pop(0)
            backend.send_multipart([worker_addr, msg[0], msg[1]])

        if frontend in socks and socks[frontend] == zmq.POLLIN:
            client_addr, request = frontend.recv_multipart()
            clients[client_addr] = 1
            while len(workers) > 0:
                worker_addr, t = workers.popitem()
                if t - now > 1:
                    continue
                backend.send_multipart([worker_addr, client_addr, request])
                break
            else:
                msg_cache.append([client_addr, request])
    
def server_worker(ip, port, port_task): #, cache
    print("server_worker: DEALER %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.DEALER)
    receiver.setsockopt(zmq.RCVTIMEO, 1000)
    receiver.connect("tcp://%s:%d"%(ip, port))
    receiver.send_multipart([b"heart", b""])

    smart_router = SmratRouter(ip, port_task, #cache,
                ECTrans_config.batch_size, ECTrans_config.batch_value)
    smart_router.start()

    while True:
        try:
            client_addr, message = receiver.recv_multipart()
        except Exception as e:
            #print(e)
            receiver.send_multipart([b"heart", b""])
            continue
        print('client:', client_addr, type(message), len(message))
        jsdat = json.loads(message)
        smart_router.send_json(jsdat)
        receiver.send_multipart([client_addr, b"world"])

def server_req(ip, port, port_task):
    # @debug
    print("server_req: REP %d ==> %d " % (port, port_task) )
    context = zmq.Context()
    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://%s:%d"%(ip, port))

    smart_router = SmratRouter(ip, port_task, ECTrans_config.batch_size, ECTrans_config.batch_value)
    smart_router.start()
    while True:
        dat = receiver.recv_json()
        smart_router.send_json(dat)
        #tid = dat['tid']
        #print('task id:', tid)
        receiver.send('OK'.encode())

def proc_encode(ip, port_task, q_enc, model_encoder):
    """
    编码器
    """
    consumer_id = random.randrange(1000,9999)
    print("proc_encode ID: #%s ==> PORT PULL:%d" % (consumer_id, port_task) )
    context = zmq.Context()
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://%s:%s"%(ip, port_task))

    while True:
        data = consumer_receiver.recv_json()
        tid = data['client_ids']
        dat = data['texts']
        batch_input = torch.LongTensor(dat).to(config.device)
        del dat
        torch.cuda.empty_cache()
        src_enc, src_mask = translate_encode(batch_input, model_encoder)
        # print('src_enc:',type(src_enc))
        # print('encoder:%s' % consumer_id)
        q_enc.put( (tid, src_enc, src_mask) )
        torch.cuda.empty_cache()

def proc_decode(q_enc, ip, port_out_pub, model_decoder, model_generator):
    # send work
    consumer_id = random.randrange(1000,9999)
    print("proc_decode ID: #%s ==> PORT PUSH:%d" % (consumer_id, port_out_pub) )
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect("tcp://%s:%d"%(ip, port_out_pub))

    while True:
        tid, dat, src_mask = q_enc.get()
        torch.cuda.empty_cache()
        src_enc = dat.clone()
        del dat
        torch.cuda.empty_cache()
        translation = translate_decode_split(src_enc, src_mask, model_decoder, model_generator, use_beam=True)
        torch.cuda.empty_cache()
        result = {'client_ids':tid, 'result':translation}
        # print('client_ids:', tid)
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
        client_ids = ret['client_ids']

        for ids in client_ids:
            # ('1630303041986551_16',(0,64))
            client_id, (b, e) = ids
            dat = ret['result'][b:e]
            packet = {'client_id': client_id, 'result':dat}
            # print('packet client_id:', client_id)
            publisher.send_json(packet)
            # print('publish dat')
            # title = str(ret)[:25]
            # print('publish:%s' % title)

def result_collector(ip, port_out, total, task_id, result_queue):
    # print("result_collector:  ==> %d " % (port_out) )
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
class ECTrans_Server():
    def __init__(self,
                ip='127.0.0.1',
                port=5550,
                port_out=5560,
                realtime_pipelines=2,
                batch_pipelines=2
                ):
        self.ip = ip
        self.port = port
        self.port_task = port + 100
        # self.port_out_encoder = port + 200 
        # self.port_in_decoder = port + 300 
        self.port_out_pub = port_out + 100
        self.port_out = port_out

        #self.workers_real = ECTrans_config.realtime_pipelines
        #self.workers_batch = ECTrans_config.batch_pipelines
        self.workers_real = realtime_pipelines
        self.workers_batch = batch_pipelines
        self.workers = self.workers_batch + self.workers_real
        print('workers:', self.workers)

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
            q_encoder = mp.Queue()
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
            if i < self.workers_real:
                port_task = self.port_task
            else:
                port_task = self.port_task + 5
                
            p_encode = mp.Process(target=proc_encode, 
                                    args=(self.ip, port_task,
                                        self.queues[i], model_encoder))

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

        print('real pub start....')
        self.real = mp.Process(target=real_pub, 
                                args=(self.ip, self.port_task+2, self.port_task))
        self.real.start()

        self.batch = mp.Process(target=real_pub, 
                                args=(self.ip, self.port_task+1, self.port_task+5))
        self.batch.start()

        print('publisher start....')
        self.publisher = mp.Process(target=result_pub, 
                                    args=(self.ip, self.port_out_pub, self.port_out))
        self.publisher.start()
        print('route start....')
        self.route = mp.Process(target=server_route, 
                                    args=(self.ip, self.port, self.port+1))
        self.route.start()

        print('server_work start....')
        for i in range(3):
            self.server_work = mp.Process(target=server_worker,  #server_req
                                    args=(self.ip, self.port+1, self.port_task)) #, self.cache
            self.server_work.start()
        
        print('ECTrans server ready....')

    def stop(self):
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

class ECTrans_Client():
    def __init__(self,
                ip='127.0.0.1',
                port=5550,
                port_out=5560):

        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.batch_size = ECTrans_config.batch_size
        self.result = []
        self.total = 0
        self.collector = None
        self.encoder = None
        self.task_id = 0

    
    def set_encoder(self, fun):
        self.encoder = fun

    def send(self, datafile):
        if self.encoder is None:
            raise ValueError("请先用set_encode设置编码方法")

        txts = readtxt(datafile)
        sentences = list(filter(None, txts.splitlines()))
        
        total = len(sentences)
        self.total = len(sentences)

        task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
        self.task_id = task_id

        result_queue = mp.Queue()
        self.collector = mp.Process(target=result_collector, 
                                    args=(self.ip, self.port_out, total, task_id, result_queue))
        self.collector.start()

        context = zmq.Context()
        #zmq_socket = context.socket(zmq.REQ)
        zmq_socket = context.socket(zmq.DEALER)
        zmq_socket.connect("tcp://%s:%d"%(self.ip, self.port))

        batch_text = self.encoder(sentences)
        # print('batch_text:', type(batch_text))
        work_message = {'tid':task_id, 'texts': batch_text, 'length': total}
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
    
    def __enter__(self):
        pass

    def __exit__(self):
        pass
        if not self.collector is None:
            if self.collector.is_alive():
                self.collector.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECTrans')
    parser.add_argument('--cmd', type=str, required=True, default="", help='server, client')
    parser.add_argument('--ip', type=str, default="127.0.0.1", help='ip')
    parser.add_argument('--port', type=int, default=5550, help='port')
    parser.add_argument('--port_out', type=int, default=5560, help='port_out')
    parser.add_argument('--realtime_pipelines', type=int, default=2, help='pipeline number')
    parser.add_argument('--batch_pipelines', type=int, default=2, help='batch pipeline')
    parser.add_argument('--datafile', type=str, default="report/data_100.txt", help='file')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    args = parser.parse_args()

    cmd = args.cmd
    ip = args.ip
    port = args.port
    port_out = args.port_out

    realtime_pipelines = args.realtime_pipelines
    batch_pipelines = args.batch_pipelines

    datafile = args.datafile
    batch_size = args.batch_size

    if cmd=='server':
        print('start server...')
        server = ECTrans_Server(ip=ip,
                                port=port,
                                port_out=port_out,
                                realtime_pipelines=realtime_pipelines,
                                batch_pipelines=batch_pipelines
                                )
        server.start()

    if cmd=='client':
        
        print('start client...')

        client = ECTrans_Client(ip=ip,
                            port=port,
                            port_out=port_out)

        txt_encode = lambda x: get_sample(x).numpy().tolist()
        client.set_encoder(txt_encode)

        print('send data...')
        sents = client.send(datafile)
        
        print('total results :%d' % len(sents))

