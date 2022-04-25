#!/usr/bin/env python3
#coding:utf-8

import os
import sys
import json
import time
import random
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5560")

channel = ''
if len(sys.argv)==2:
    channel = sys.argv[1]

filter_title = ''
if channel:
    # filter_title = json.dumps({'id':channel})[:-2]
    # filter_title = "{'id':'%s" % channel
    filter_title = "{\"id\":\"" + channel
    # filter_title = channel

socket.setsockopt(zmq.SUBSCRIBE, filter_title.encode())
print('SUBSCRIBE:%s'% filter_title)
print('waitting message...')

while True:
    dat = socket.recv()
    #dat = socket.recv_json()
    #print(dat)
    print(dat[:25])
    print('-'*40)


if __name__ == '__main__':
    pass

