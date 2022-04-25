#!/usr/bin/env python3
#coding:utf-8

import time
import utils
import config
import logging
import numpy as np
from pynvml import *

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from train import translate
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing
import torch.multiprocessing as mp

def one_sentence_translate(sent, beam_search=True, online=0):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    if online:
        while 1:
            try:
                print('-'*40)
                sent = input('input sentences:').strip()
            except :
                sent = ''
            if sent  in [''] : continue
            if sent in ['Q', 'q'] : break

            src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
            batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
            ret = translate(batch_input, model, use_beam=beam_search)
            print(ret)
    else:
        src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
        batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
        ret = translate(batch_input, model, use_beam=beam_search)
        print(ret)

def translate_example():
    sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
           "to childless workers."
    one_sentence_translate(sent, beam_search=True)

def online_translate():
    config.beam_size = 4
    one_sentence_translate('', beam_search=True, online=1)
#-----------------------------------------
def get_one_sample(sent):
    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    return np.array(src_tokens)

def get_sample(sents):
    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()
    PAD = english_tokenizer_load().pad_id()
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS] for sent in sents]
    ret = pad_sequence([torch.from_numpy(np.array(x)) for x in src_tokens], 
                        batch_first=True, padding_value=PAD)
    return ret

def translate_one_sample(txt, model, beam_search=True):
    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()
    sample = [[BOS] + english_tokenizer_load().EncodeAsIds(txt.strip()) + [EOS]]
    batch_input = torch.LongTensor(np.array(sample)).to(config.device)
    ret = translate(batch_input, model, use_beam=beam_search)
    #print('translate:', ret)
    return ret

def translate_texts(sentences, model, beam_search=True):
    result = []
    for txt in sentences:
        if txt:
            ret = translate_one_sample(txt, model, beam_search=beam_search)
            result.append(ret)
    return result

def translate_batch(sentences, model, beam_search=True, batch_size=64):
    total = len(sentences)
    #if total > batch_size:
    sentlist = [sentences[i*batch_size:(i+1)*batch_size] for i in range(np.ceil(total/batch_size).astype(int))]
    result = []
    torch.cuda.empty_cache()
     
    for txts in sentlist:
        batch_dat = get_sample(txts)
        batch_input = torch.LongTensor(batch_dat).to(config.device)
        ret = translate(batch_input, model, use_beam=beam_search)
        #print('translate:\n', '\n'.join(ret))
        result.extend(ret)
    torch.cuda.empty_cache()
    return result

def readtxt(fname, encoding='utf-8'):
    try:
        with open(fname, 'r', encoding=encoding) as f:  
            data = f.read()
        return data
    except Exception as e:
        return ''

def savetofile(txt, filename, encoding='utf-8', method='a+'):
    try:
        with open(filename, method, encoding=encoding) as f:  
            f.write(str(txt)+ '\n')
        return 1
    except :
        return 0


def GPU_memory(gpuid=0):
    NUM_EXPAND = 1024 * 1024
    handle = nvmlDeviceGetHandleByIndex(gpuid)
    info = nvmlDeviceGetMemoryInfo(handle)

    gpu_memory_used = info.used / NUM_EXPAND
    #print('Total Memory:%d MB,  Used Memory:%d MB'% (gpu_memory_total, gpu_memory_used))
    return  gpu_memory_used

class GPU_MEM():
    def __init__(self, gupid=0, interval=1):
        self.gupid = gupid
        self.interval = interval
        self.status = 0
        self.data = []
        self.queue = mp.Queue()
        self.process = None

    def get_gpu_memory(self):
        while True:
            mem = GPU_memory(self.gupid)
            #print('memory:', mem)
            self.queue.put(mem)
            if self.interval>0:
                time.sleep(self.interval)
            else:
                break;
        
    def build(self):
        pass
        self.data = []
        self.process = mp.Process(target=self.get_gpu_memory)

    def start(self, interval=1):
        self.interval = interval
        if not self.process is None:
            self.process.start()
            #self.process.join()
            
    def stop(self):
        self.interval = 0
        #self.process.stop()
        self.process.terminate()
        self.data = self.get_queue()

    def mem_ave(self):
        if self.data == []:
            ret = 0
        else:
            ret = np.average(self.data)
        
        return ret
    
    def mem_max(self):
        if self.data == []:
            ret = 0
        else:
            ret = np.max(self.data)
       
        return ret

    def get_queue(self):
        ret = []
        while not self.queue.empty():
            dat = self.queue.get()
            ret.append(dat)
        return ret

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    #translate_example()
    online_translate()


