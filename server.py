#!/usr/bin/env python3
#coding:utf-8

import argparse
import os
import re
import json
import logging
from saas_lib import *
from flask import Flask, request, render_template, jsonify, json

import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from train import translate
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='server')
parser.add_argument('--ip', type=str, default="0.0.0.0", help='IP')
parser.add_argument('--port', type=int, default=8090, help='port,default:8090')
parser.add_argument('--debug', type=int, default=1, help='debug')
args = parser.parse_args()

app = Flask(__name__)

def get_sample(sent):
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    return batch_input

model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                   config.d_model, config.d_ff, config.n_heads, config.dropout)


@app.route('/')
def index():
    datlist = svclist.jsondat()
    dat = [ list(x.values()) for x in datlist]
    print(dat)
    return render_template('index.html', svclist=dat, version=gblVersion)

#@app.route('/trans/<txt>')
#def trans(txt):
@app.route('/trans', methods=['POST'])
def trans():
    res = {}
    txt = request.values.get('appid', '')
    if not txt:
        res["result"] = "Error"
        return jsonify(res)

    print('text:', txt)
    batch_input = get_sample(txt)
    ret = translate(batch_input, model, use_beam=beam_search)
    print('translate:', ret)
    res["result"] = 'OK'
    res["dat"] = ret
    return jsonify(res)

if os.name=='nt':
    logging.info('Running under windows...')
    logging.info('service is ready...')
    app.run(
        host = args.ip,
        port = args.port,
        debug = bool(args.debug)
    )
else:
    # start server for linux 
    from gevent import pywsgi
    logging.info('Running under Linux...')
    logging.info('service is ready...')
    #app.debug = bool(args.debug)
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    server.serve_forever()


if __name__ == '__main__':
    pass
