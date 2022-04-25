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

from ECTrans_task import *


def dotask(datafile, embedding=0, batch=0):
    client = ECTrans_Client(ip="127.0.0.1",
                        port=5588,
                        port_out=5560,
                        batch_size=1)

    # 开始计时
    # start = time.time()
    if batch:
        sents = client.send_batch(datafile, embedding=embedding)
    else:
        sents = client.send(datafile, embedding=embedding)

    tid = client.task_id
    now = time.time()
    total = len(sents)
    print('task id:%s, total:%d, finished_time:%s' % (tid, total, now))

def run_test(task_list, embedding=0, batch=0):
    for tasks in task_list:
        logging.info(('test :%s'%str(tasks)).center(60, '-'))
        start = time.time()
        task_process = []
        for num, fn in tasks:
            datfile = './report/data_%s.txt' % fn
            for i in range(num):
                process = mp.Process(target=dotask, args=(datfile,embedding, batch))                 
                task_process.append(process)
    
        for p in task_process:
            p.start()
            #p.join()
            #time.sleep(0.01)

        #for p in task_process:
        #    p.join()

        predict_time = (time.time() - start)*1000
        logging.info('total time:%.2f ms' % predict_time)


def test_package():
    # 任务列表，(客户端个数，"文件名")
    task_list = [
                    [(2, '64')],
                    [(4, '64')],
                    [(8, '64')],
                    [(16, '64')]
                ]

    logging.info('test'.center(40,'='))
    run_test(task_list)

def test_single(embedding=0):
    task_list = [[(1, '1k')]]

    logging.info('实时任务测试'.center(40,'='))
    run_test(task_list, embedding=embedding)

def test_real(batch=0):
    task_list = [
                    [(16, '8')],
                    [(32, '8')],
                    [(64, '8')],
                    [(128, '8')]
                ]
    '''
    task_list = [
                    [(16, '1k')],
                    [(32, '1k')],
                    [(64, '1k')],
                    [(128, '1k')]
                ]
    '''

    logging.info('real time'.center(40,'='))
    run_test(task_list, batch=batch)

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
    task_list = [
                    [(4, '128'), (8, '32')],
                ]

    task_list = [[(8, '32'), (4, '128')]]
    task_list = [[(4, '128'), (1, '20'), (1, '100'), (1, '8'), (1, '200'), (1, '50')]]
    task_list = [[(3, '256'), (1, '16'), (1, '256'), (1, '128'), (2, '256'), (1, '16'), (1, '128')]]

    logging.info('混合任务测试'.center(40,'='))
    run_test(task_list)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECTrans_Task')
    parser.add_argument('--cmd', type=str, required=True, default="", help='real, mix')
    parser.add_argument('--log', type=str, default="ECTrans_task_test.log", help='log')
    parser.add_argument('--embedding', type=int, default=0, help='')
    parser.add_argument('--batch', type=int, default=0, help='')
    args = parser.parse_args()

    cmd = args.cmd
    embedding = args.embedding
    batch = args.batch
    log = args.log


    fmttxt = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    ft = os.path.splitext(log)
    log = '%s_%s%s' % (ft[0], fmttxt, ft[1])

    #################################################################################################
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', log),
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

    if cmd == 'single':
        test_single(embedding=embedding)
    if cmd == 'real':
        test_real(batch=batch)
    if cmd == 'package':
        test_package()
    if cmd == 'mix':
        test_mix()
    
    
