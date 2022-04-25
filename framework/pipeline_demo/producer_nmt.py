#!/usr/bin/env python3
#coding:utf-8

import os
import sys
sys.path.append('../../')

import time
import zmq
import random
from tqdm import tqdm
from translate import *
 
def producer(ip, port, datafile):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://%s:%d"%(ip, port) )

    txts = readtxt(datafile)
    sentences = list(filter(None, txts.splitlines()))
    batch_size = 8
    total = len(sentences)
    #if total > batch_size:
    sentlist = [sentences[i*batch_size:(i+1)*batch_size] for i in range(np.ceil(total/batch_size).astype(int))]
    #else:
    #    sentlist = [sentences]

    total_batch = len(sentlist)
    work_id = random.randrange(1000, 9999)
    print('total:', total)
    print('total_batch:', total_batch)

    msg = input('press enter to start:')
    for i in tqdm(range(total_batch), ncols=80):
        wid = '%d_%d' % (work_id, i)
        work_message = {'id': wid, 'texts':sentlist[i]}
        zmq_socket.send_json(work_message)
        if i % 100 == 0:
            time.sleep(0.1)
 
if __name__ == '__main__':
    pass

    datafile = '../../data_100.txt'
    if len(sys.argv)==2:
        datafile = sys.argv[1]

    producer('127.0.0.1', 5557, datafile)
