#!/usr/bin/env python3
#coding:utf-8

import os
import sys
import time
import zmq
import random
from tqdm import tqdm

import torch.multiprocessing as mp
from resultcollector import *
from consumer import *

def producer(ip, port):


    #-----------------------------------------
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://%s:%s"%(ip, port) )
    msg = input('press enter to start:')

    result = {}
    total = 1000
    collector = mp.Process(target=result_collector, args=('127.0.0.1', 5558, total, result) )
    collector.start()

    for i in tqdm(range(2000), ncols=80):
        num = random.randrange(10,100)
        work_message = {'num' : num}
        zmq_socket.send_json(work_message)
        if num % 100 == 0:
            time.sleep(0.1)
    
    collector.join()
    print('result:', result)

if __name__ == '__main__':
    pass
    producer('127.0.0.1', 5557)
