#!/usr/bin/env python3
#coding:utf-8

import os
import sys
import json
import time
import random
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5560")
task_id = int(time.time()*1000)*1000 + random.randrange(1000, 9999)
print('task_id:', task_id)

while True:
    msg = input('input your data:')
    if msg in ['q','Q', 'quit', 'Quit']:
        break;
    if msg:
        #socket.send(msg.encode())
        wid = '%d_%d' % (task_id, 123)
        message = {'id':wid, 'texts': msg}
        socket.send_json(message)


if __name__ == '__main__':
    pass

