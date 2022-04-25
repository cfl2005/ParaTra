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
from ECTrans_task_hrrn import *

def dotask(datafile):
    #print('正在启动客户端...')
    client = ECTrans_Client(ip="127.0.0.1",
                        port=6588,
                        port_out=6500,
                        batch_size=1)

    # 开始计时
    # start = time.time()
    # print('正在发送数据...')
    sents = client.send(datafile)
    tid = client.task_id
    now = time.time()
    total = len(sents)
    print('task id:%s, total:%d, finished_time:%s' % (tid, total, now))

def run_test(task_list):

    for tasks in task_list:
        logging.info(('batch test :%s'%str(tasks)).center(60, '-'))
        # 开始计时
        start = time.time()
        task_process = []
        for num, fn in tasks:
            datfile = './report/data_%s.txt' % fn
            for i in range(num):
                process = mp.Process(target=dotask, args=(datfile,))
                task_process.append(process)
    
        for p in task_process:
            p.start()
            time.sleep(0.1)

        for p in task_process:
            p.join()

        predict_time = (time.time() - start)*1000
        logging.info('total time:%.2f ms' % predict_time)

def test_package():
    task_list = [
                    [(2, '64')],
                    [(4, '64')],
                    [(8, '64')],
                    [(16, '64')]
                ]

    logging.info('package test'.center(40,'='))
    run_test(task_list)

def test_real():
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

    logging.info('real time test'.center(40,'='))
    run_test(task_list)

def test_mix():
    task_list = [
                    [(4, '128'), (8, '32')],
                    [(4, '128'), (24, '32')],
                    [(4, '128'), (48, '32')],
                ]
    task_list = [[(4, '32'), (4, '128')]]
    task_list = [[(4, '128'), (8, '32')]]
    task_list = [[(4, '128'), (1, '20'), (1, '100'), (1, '8'), (1, '200'), (1, '50')]]
    task_list = [[(3, '256'), (1, '16'), (1, '256'), (1, '128'), (2, '256'), (1, '16'), (1, '128')]]
    task_list = [[(3, '256'), (2, '16'), (2, '256'), (3, '128'), (3, '16'), (1, '128')]]

    logging.info('混合任务测试'.center(40,'='))
    run_test(task_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECTrans_Task_hrrn框架测试')
    parser.add_argument('--cmd', type=str, required=True, default="", help='启动方式: real, mix')
    parser.add_argument('--log', type=str, default="ECTrans_task_hrrn_test.log", help='日志文件')
    args = parser.parse_args()

    cmd = args.cmd
    log = args.log
    fmttxt = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    ft = os.path.splitext(log)
    log = '%s_%s%s' % (ft[0], fmttxt, ft[1])

    #################################################################################################
    # 指定日志
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


    if cmd == 'real':
        test_real()
    if cmd == 'package':
        test_package()
    if cmd == 'mix':
        test_mix()