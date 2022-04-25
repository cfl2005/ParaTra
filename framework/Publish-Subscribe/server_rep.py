#!/usr/bin/env python3
#coding:utf-8

import os
import sys
import json
import time
import random
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5560")
print('tcp://127.0.0.1:5560')

while True:
    dat = socket.recv()
    print(dat)
    socket.send(dat)
    print('-'*40)



if __name__ == '__main__':
    pass
