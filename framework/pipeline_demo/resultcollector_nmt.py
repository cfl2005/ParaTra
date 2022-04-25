#!/usr/bin/env python3
#coding:utf-8

import os
# sys.path.append('../../')

import sys
import time
import zmq
import pprint
 
def result_collector(ip, port_out, total):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://%s:%d"%(ip, port_out))

    collecter_data = {}
    total_result = 0
       
    #for x in range(1000):
    while True:
        result = receiver.recv_json()

        sents = result['result']
        t_sents = len(sents)

        wid = result['id']
        total_result += t_sents

        cid = result['consumer']
        cons = 'work_%d' % cid
        if cons in collecter_data.keys():
            collecter_data[cons] += t_sents
        else:
            collecter_data[cons] = t_sents
    
        if total_result >= total:
            break
    pprint.pprint(collecter_data)

 
if __name__ == '__main__':
    pass
    total = 100
    if len(sys.argv)==2:
        total = int(sys.argv[1])
    
    result_collector('127.0.0.1', 5558, total)

