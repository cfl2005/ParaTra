#!/usr/bin/env python3
#coding:utf-8

'''
import argparse
import os
import sys
import time
import torch.multiprocessing as mp
'''
import logging

from ECTrans_framework import *

def dotask(datafile):
    client = ECTrans_Client(ip="127.0.0.1",
                        port=5550,
                        port_out=5560)

    txt_encode = lambda x: get_sample(x).numpy().tolist()
    client.set_encoder(txt_encode)
    start = time.time()
    sents = client.send(datafile)
    tid = client.task_id

    total = len(sents)

    predict_time = (time.time() - start)*1000
    avetime = predict_time/total
    result = []
    #result.append('-'*40 )
    logging.info('\t'.join(result) ) #+ '\n'
    

def run_test(task_list):

    for tasks in task_list:
        start = time.time()
        task_process = []
        for num, fn in tasks:
            datfile = './report/data_%s.txt' % fn
            for i in range(num):
                process = mp.Process(target=dotask, args=(datfile,))                 
                task_process.append(process)
    
        for p in task_process:
            p.start()
        for p in task_process:
            p.join()

        predict_time = (time.time() - start)*1000
        time.sleep(2)

def test_real():
    task_list = [
                    [(16, '8')],
                    [(32, '8')],
                    [(64, '8')],
                    [(128, '8')]
                ]
    run_test(task_list)

def test_mix():
    '''
    task_list = [
                    [(4, '1k'), (8, '8')],
                    [(4, '1k'), (24, '8')],
                    [(4, '1k'), (48, '8')],
                ]
    '''
    task_list = [
                    [(4, '128'), (8, '32')],
                    [(4, '128'), (24, '32')],
                    [(4, '128'), (48, '32')],
                ]
    run_test(task_list)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECTrans test')
    parser.add_argument('--cmd', type=str, required=True, default="", help='start: real, mix')
    args = parser.parse_args()

    cmd = args.cmd

    #################################################################################################
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', 'ECTrans_batch_test.log'  ),
                filemode='a'
                )
    #################################################################################################
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    #formatter = logging.Formatter('[%(asctime)s]%(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################


    if cmd == 'real':
        test_real()
    if cmd == 'mix':
        test_mix()
    
    
