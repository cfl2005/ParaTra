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

import utils
import config
from train import translate, translate_encode, translate_decode, translate_decode_split
from model import *
from translate import *

import framework.pipeline_demo.ECTrans_framework.ECTrans_client #import *


parser = argparse.ArgumentParser(description='ECTrans client')
parser.add_argument('--datafile', type=str, default="report/data_100.txt", help='file')
args = parser.parse_args()
datafile = args.datafile

# python3 ECTrans.py --cmd=client --batch_size=128 --datafile=data_100.txt
# python3 ECTrans.py --cmd=client --batch_size=8 --datafile=data_20.txt

print('start client...')

client = ECTrans_Client(ip='127.0.0.1',
                    port=5557,
                    port_out=5560,)


txt_encode = lambda x: get_sample(x).numpy().tolist()
client.set_encoder(txt_encode)

print('send data...')
sents = client.send(datafile)
print('wait result...')
print('total results :%d' % len(sents))



if __name__ == '__main__':
    pass

